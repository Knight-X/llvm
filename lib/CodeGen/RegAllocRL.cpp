//===- RegAllocBase.cpp - Register Allocator Base Class -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the RegAllocBase class which provides common functionality
// for LiveIntervalUnion-based register allocators.
//
//===----------------------------------------------------------------------===//

#include "RegAllocRL.h"
#include "Spiller.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LiveInterval.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveRegMatrix.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>

using namespace llvm;

#define DEBUG_TYPE "regalloc"

STATISTIC(NumNewQueued    , "Number of new live ranges queued");

// Temporary verification option until we can put verification inside
// MachineVerifier.

const char RegAllocRL::TimerGroupName[] = "regalloc";
const char RegAllocRL::TimerGroupDescription[] = "Register Allocation";
bool RegAllocRL::VerifyEnabled = false;

bool RegAllocRL::terminalState = false;
bool RegAllocRL::initialState = true;
bool RegAllocRL::inference = false;
std::vector<float> RegAllocRL::_state(256, 0);
long int RegAllocRL::_score = 0;
float RegAllocRL::prev_weight = std::numeric_limits<float>::max();
float RegAllocRL::curr_weight = 0.0;
QTable RegAllocRL::g = QTable();

//===----------------------------------------------------------------------===//
//                         RegAllocBase Implementation
//===----------------------------------------------------------------------===//

// Pin the vtable to this file.
void RegAllocRL::anchor() {}

void RegAllocRL::init(VirtRegMap &vrm,
                        LiveIntervals &lis,
                        LiveRegMatrix &mat) {
  TRI = &vrm.getTargetRegInfo();
  MRI = &vrm.getRegInfo();
  VRM = &vrm;
  LIS = &lis;
  Matrix = &mat;
  MRI->freezeReservedRegs(vrm.getMachineFunction());
  RegClassInfo.runOnMachineFunction(vrm.getMachineFunction());
  learn = new QLearner(&g, 0.001, 0.1);
  q = new GreedyQ(&g);
  policy = new EpsilonPolicy(q, &r, 0.01);
}

// Visit all the live registers. If they are already assigned to a physical
// register, unify them with the corresponding LiveIntervalUnion, otherwise push
// them on the priority queue for later assignment.
void RegAllocRL::seedLiveRegs() {
  NamedRegionTimer T("seed", "Seed Live Regs", TimerGroupName,
                     TimerGroupDescription, TimePassesIsEnabled);
  for (unsigned i = 0, e = MRI->getNumVirtRegs(); i != e; ++i) {
    unsigned Reg = TargetRegisterInfo::index2VirtReg(i);
    if (MRI->reg_nodbg_empty(Reg))
      continue;
    enqueue(&LIS->getInterval(Reg));
  }
}


int RegAllocRL::calculateReward(unsigned action, float weight) {
  int reward = -1;
  
  if (weight >= 0.0) {
    curr_weight += weight;
        if (curr_weight >= prev_weight * 1.2 && terminalState) {
	  reward = -10000;
	} else if (curr_weight >= prev_weight * 1.5 && terminalState) {
	  reward = -20000;
	} else if (curr_weight < prev_weight && terminalState) {
		reward = 10000;
	}
  } else {
    reward -= 5;
  }
  _score += reward;
  initialState = false;
  std::cout << "weight: " << weight << std::endl;
  std::cout << "prev_weight: " << prev_weight << std::endl;
  std::cout << "curr_weight: " << curr_weight << std::endl;
  std::cout << "score: " << _score << std::endl;
  return reward;
}
void RegAllocRL::observe(std::vector<float> old_state, unsigned action, float reward, std::vector<float> new_state) {
  if (old_state.size() != 256 || new_state.size() != 256) {
        report_fatal_error("wrong size");
  }
  learn->observe(old_state, action, reward, new_state, past_cand);
}

// Top-level driver to manage the queue of unassigned VirtRegs and call the
// selectOrSplit implementation.
void RegAllocRL::allocatePhysRegs() {
  seedLiveRegs();
  unsigned prev_action = 0;
  int prev_reward = 0;

  // Continue assigning vregs one at a time to available physical registers.
  while (LiveInterval *VirtReg = dequeue()) {
    assert(!VRM->hasPhys(VirtReg->reg) && "Register already assigned");

    // Unused registers can appear when the spiller coalesces snippets.
    if (MRI->reg_nodbg_empty(VirtReg->reg)) {
      DEBUG(dbgs() << "Dropping unused " << *VirtReg << '\n');
      aboutToRemoveInterval(*VirtReg);
      LIS->removeInterval(VirtReg->reg);
      continue;
    }

    // Invalidate all interference queries, live ranges could have changed.
    Matrix->invalidateVirtRegs();

    // selectOrSplit requests the allocator to return an available physical
    // register if possible and populate a list of new live intervals that
    // result from splitting.
    DEBUG(dbgs() << "\nselectOrSplit "
          << TRI->getRegClassName(MRI->getRegClass(VirtReg->reg))
          << ':' << *VirtReg << " w=" << VirtReg->weight << '\n');

    using VirtRegVec = SmallVector<unsigned, 4>;

    VirtRegVec SplitVRegs;
    float weight = 0.0;
    unsigned AvailablePhysReg = selectOrSplit(*VirtReg, SplitVRegs, prev_action, prev_reward, weight);

    if (AvailablePhysReg == ~0u) {
      // selectOrSplit failed to find a register!
      // Probably caused by an inline asm.
      MachineInstr *MI = nullptr;
      for (MachineRegisterInfo::reg_instr_iterator
           I = MRI->reg_instr_begin(VirtReg->reg), E = MRI->reg_instr_end();
           I != E; ) {
        MachineInstr *TmpMI = &*(I++);
        if (TmpMI->isInlineAsm()) {
          MI = TmpMI;
          break;
        }
      }
      if (MI)
        MI->emitError("inline assembly requires more registers than available");
      else
        report_fatal_error("ran out of registers during register allocation");
      // Keep going after reporting the error.
      VRM->assignVirt2Phys(VirtReg->reg,
                 RegClassInfo.getOrder(MRI->getRegClass(VirtReg->reg)).front());
      continue;
    }

    if (AvailablePhysReg) {
      Matrix->assign(*VirtReg, AvailablePhysReg);
      prev_reward = calculateReward(AvailablePhysReg, weight);
      prev_action = AvailablePhysReg;
    }

    for (unsigned Reg : SplitVRegs) {
      assert(LIS->hasInterval(Reg));

      LiveInterval *SplitVirtReg = &LIS->getInterval(Reg);
      assert(!VRM->hasPhys(SplitVirtReg->reg) && "Register already assigned");
      if (MRI->reg_nodbg_empty(SplitVirtReg->reg)) {
        assert(SplitVirtReg->empty() && "Non-empty but used interval");
        DEBUG(dbgs() << "not queueing unused  " << *SplitVirtReg << '\n');
        aboutToRemoveInterval(*SplitVirtReg);
        LIS->removeInterval(SplitVirtReg->reg);
        continue;
      }
      DEBUG(dbgs() << "queuing new interval: " << *SplitVirtReg << "\n");
      assert(TargetRegisterInfo::isVirtualRegister(SplitVirtReg->reg) &&
             "expect split value in virtual register");
      enqueue(SplitVirtReg);
      ++NumNewQueued;
    }
  }
  terminalState = true;
  prev_reward = calculateReward(prev_action, 0.0);
  if (!inference) {
  observe(_state, prev_action, prev_reward, std::vector<float>(256, 0));
  }
}

void RegAllocRL::postOptimization() {
  spiller().postOptimization();
  for (auto DeadInst : DeadRemats) {
    LIS->RemoveMachineInstrFromMaps(*DeadInst);
    DeadInst->eraseFromParent();
  }
  DeadRemats.clear();
}
