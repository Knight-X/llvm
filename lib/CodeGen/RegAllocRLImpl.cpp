//===-- RegAllocBasic.cpp - Basic Register Allocator ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the RABasic function pass, which provides a minimal
// implementation of the basic register allocator.
//
//===----------------------------------------------------------------------===//

#include "AllocationOrder.h"
#include "LiveDebugVariables.h"
#include "RegAllocRL.h"
#include "Spiller.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/CalcSpillWeights.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveRangeEdit.h"
#include "llvm/CodeGen/LiveRegMatrix.h"
#include "llvm/CodeGen/LiveStacks.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/PassAnalysisSupport.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <queue>
#include <time.h>
#include <stdlib.h>
#include <fstream>

using namespace llvm;

#define DEBUG_TYPE "regalloc"

static RegisterRegAlloc rlRegAlloc("rl", "rl register allocator",
                                      createRLRegisterAllocator);

namespace {
  struct CompSpillWeight {
    bool operator()(LiveInterval *A, LiveInterval *B) const {
      return A->weight < B->weight;
    }
  };
}

typedef std::map<std::pair<std::vector<float>, int>, float> stringMap;
bool mapToFile(const std::string &filename,const stringMap &fileMap)     //Write Map
{
    std::ofstream ofile;
    ofile.open(filename.c_str());
    if(!ofile)
    {
        return false;           //file does not exist and cannot be created.
    }

    ofile << RegAllocRL::prev_weight << "\n";
    ofile << RegAllocRL::curr_weight << "\n";
    for(stringMap::const_iterator iter= fileMap.begin(); iter!=fileMap.end(); ++iter)
    {

	std::pair<std::vector<float>, int> first = iter->first;
	
	for (int i = 0; i < 256; i++) {
	  ofile << first.first[i] << "&";  
	}
	ofile << first.second;
	ofile<<"&"<<iter->second;
        ofile<<"\n";
    }
    return true;
}
void splitString(std::vector<float> &v_str,const std::string &str,const char ch, int& reg, float& val)
{
	std::string sub;
	std::string::size_type pos = 0;
	std::string::size_type old_pos = 0;
	bool flag=true;
			     
	int i = 0;
	while(flag)
	{
		pos=str.find_first_of(ch,pos);
		if(pos == std::string::npos)
		{
			flag = false;
			pos = str.size();
		}
		std::string::size_type sz;
		sub = str.substr(old_pos,pos-old_pos);  // Disregard the '.'
		if (i < 256) {
		float ret = std::atof(sub.c_str());
		v_str.push_back(ret);
		} else if (i == 256){
		int ret = std::stoi(sub, &sz);
		  reg = ret;
		} else {
		  int ret = std::atof(sub.c_str());
		  val = ret;
		}
		old_pos = ++pos;
		i++;
	}
}
bool fileToMap(const std::string &filename, stringMap &fileMap)  //Read Map
{
	std::ifstream ifile;
    ifile.open(filename.c_str());
    if(!ifile)
        return false;   //could not read the file.
    std::string line;
    std::string key;
    ifile >> line;
    std::string::size_type sz;
    RegAllocRL::prev_weight = std::stof(line, &sz);
    while(ifile>>line)
    {
        std::vector<float> v_str;
        int reg = 0;
        float val = 0.0;
        splitString(v_str,line,'&', reg, val);
        std::pair<std::vector<float>, int> a(v_str, reg);
	fileMap[a] = val;
    }
    return true;
}

namespace {
/// RABasic provides a minimal implementation of the basic register allocation
/// algorithm. It prioritizes live virtual registers by spill weight and spills
/// whenever a register is unavailable. This is not practical in production but
/// provides a useful baseline both for measuring other allocators and comparing
/// the speed of the basic algorithm against other styles of allocators.
class RARL : public MachineFunctionPass,
                public RegAllocRL,
                private LiveRangeEdit::Delegate {
  // context
  MachineFunction *MF;

  // state
  std::unique_ptr<Spiller> SpillerInstance;
  std::priority_queue<LiveInterval*, std::vector<LiveInterval*>,
                      CompSpillWeight> Queue;

  // Scratch space.  Allocated here to avoid repeated malloc calls in
  // selectOrSplit().
  BitVector UsableRegs;

  bool LRE_CanEraseVirtReg(unsigned) override;
  void LRE_WillShrinkVirtReg(unsigned) override;
  SmallVector<unsigned, 8> past_cand;
public:
  RARL();

  /// Return the pass name.
  StringRef getPassName() const override { return "Basic Register Allocator"; }

  /// RABasic analysis usage.
  void getAnalysisUsage(AnalysisUsage &AU) const override;

  void releaseMemory() override;

  Spiller &spiller() override { return *SpillerInstance; }

  void enqueue(LiveInterval *LI) override {
    Queue.push(LI);
  }

  LiveInterval *dequeue() override {
    if (Queue.empty())
      return nullptr;
    LiveInterval *LI = Queue.top();
    Queue.pop();
    return LI;
  }

  unsigned selectOrSplit(LiveInterval &VirtReg,
                         SmallVectorImpl<unsigned> &SplitVRegs,
			 unsigned prev_action,
			 int prev_reward,
			 float& weight) override;

  std::vector<float> get(SmallVector<unsigned, 8> Cands, LiveInterval &VirtReg);
  unsigned pickAction(std::vector<float> state, SmallVector<unsigned, 8>& cand);
  void observe(std::vector<float> old_state, unsigned action, float reward, std::vector<float> new_state);
  /// Perform register allocation.
  bool runOnMachineFunction(MachineFunction &mf) override;

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::NoPHIs);
  }

  // Helper for spilling all live virtual registers currently unified under preg
  // that interfere with the most recently queried lvr.  Return true if spilling
  // was successful, and append any new spilled/split intervals to splitLVRs.
  bool spillInterferences(LiveInterval &VirtReg, unsigned PhysReg,
                          SmallVectorImpl<unsigned> &SplitVRegs);

  static char ID;
};

char RARL::ID = 0;

} // end anonymous namespace

char &llvm::RARLID = RARL::ID;

INITIALIZE_PASS_BEGIN(RARL, "regallocrl", "RL Register Allocator",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(LiveDebugVariables)
INITIALIZE_PASS_DEPENDENCY(SlotIndexes)
INITIALIZE_PASS_DEPENDENCY(LiveIntervals)
INITIALIZE_PASS_DEPENDENCY(RegisterCoalescer)
INITIALIZE_PASS_DEPENDENCY(MachineScheduler)
INITIALIZE_PASS_DEPENDENCY(LiveStacks)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTree)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfo)
INITIALIZE_PASS_DEPENDENCY(VirtRegMap)
INITIALIZE_PASS_DEPENDENCY(LiveRegMatrix)
INITIALIZE_PASS_END(RARL, "regallocrl", "RL Register Allocator", false,
                    false)

bool RARL::LRE_CanEraseVirtReg(unsigned VirtReg) {
  LiveInterval &LI = LIS->getInterval(VirtReg);
  if (VRM->hasPhys(VirtReg)) {
    Matrix->unassign(LI);
    aboutToRemoveInterval(LI);
    return true;
  }
  // Unassigned virtreg is probably in the priority queue.
  // RegAllocBase will erase it after dequeueing.
  // Nonetheless, clear the live-range so that the debug
  // dump will show the right state for that VirtReg.
  LI.clear();
  return false;
}

void RARL::LRE_WillShrinkVirtReg(unsigned VirtReg) {
  if (!VRM->hasPhys(VirtReg))
    return;

  // Register is assigned, put it back on the queue for reassignment.
  LiveInterval &LI = LIS->getInterval(VirtReg);
  Matrix->unassign(LI);
  enqueue(&LI);
}

RARL::RARL(): MachineFunctionPass(ID) {
}

void RARL::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addRequired<AAResultsWrapperPass>();
  AU.addPreserved<AAResultsWrapperPass>();
  AU.addRequired<LiveIntervals>();
  AU.addPreserved<LiveIntervals>();
  AU.addPreserved<SlotIndexes>();
  AU.addRequired<LiveDebugVariables>();
  AU.addPreserved<LiveDebugVariables>();
  AU.addRequired<LiveStacks>();
  AU.addPreserved<LiveStacks>();
  AU.addRequired<MachineBlockFrequencyInfo>();
  AU.addPreserved<MachineBlockFrequencyInfo>();
  AU.addRequiredID(MachineDominatorsID);
  AU.addPreservedID(MachineDominatorsID);
  AU.addRequired<MachineLoopInfo>();
  AU.addPreserved<MachineLoopInfo>();
  AU.addRequired<VirtRegMap>();
  AU.addPreserved<VirtRegMap>();
  AU.addRequired<LiveRegMatrix>();
  AU.addPreserved<LiveRegMatrix>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

void RARL::releaseMemory() {
  SpillerInstance.reset();
}


// Spill or split all live virtual registers currently unified under PhysReg
// that interfere with VirtReg. The newly spilled or split live intervals are
// returned by appending them to SplitVRegs.
bool RARL::spillInterferences(LiveInterval &VirtReg, unsigned PhysReg,
                                 SmallVectorImpl<unsigned> &SplitVRegs) {
  // Record each interference and determine if all are spillable before mutating
  // either the union or live intervals.
  SmallVector<LiveInterval*, 8> Intfs;

  // Collect interferences assigned to any alias of the physical register.
  for (MCRegUnitIterator Units(PhysReg, TRI); Units.isValid(); ++Units) {
    LiveIntervalUnion::Query &Q = Matrix->query(VirtReg, *Units);
    Q.collectInterferingVRegs();
    for (unsigned i = Q.interferingVRegs().size(); i; --i) {
      LiveInterval *Intf = Q.interferingVRegs()[i - 1];
      if (!Intf->isSpillable()) {
	 std::cout << "can not spill " << std::endl;
        return false;
      }
      Intfs.push_back(Intf);
    }
  }
  DEBUG(dbgs() << "spilling " << printReg(PhysReg, TRI)
               << " interferences with " << VirtReg << "\n");
  assert(!Intfs.empty() && "expected interference");

  // Spill each interfering vreg allocated to PhysReg or an alias.
  for (unsigned i = 0, e = Intfs.size(); i != e; ++i) {
    LiveInterval &Spill = *Intfs[i];

    // Skip duplicates.
    if (!VRM->hasPhys(Spill.reg))
      continue;

    // Deallocate the interfering vreg by removing it from the union.
    // A LiveInterval instance may not be in a union during modification!
    Matrix->unassign(Spill);

    // Spill the extracted interval.
    LiveRangeEdit LRE(&Spill, SplitVRegs, *MF, *LIS, VRM, this, &DeadRemats);
    spiller().spill(LRE);
  }
  return true;
}


std::vector<float> RARL::get(SmallVector<unsigned, 8> Cands, LiveInterval &VirtReg) {
  std::vector<float> ret(256, 0);
  for (SmallVectorImpl<unsigned>::iterator PhysRegI = Cands.begin(),
       PhysRegE = Cands.end(); PhysRegI != PhysRegE; ++PhysRegI) {
    unsigned PhysReg = *PhysRegI;
    bool spillable = true;
    for (MCRegUnitIterator Units(PhysReg, TRI); Units.isValid(); ++Units) {
      if (spillable) {
        LiveIntervalUnion::Query &Q = Matrix->query(VirtReg, *Units);
        Q.collectInterferingVRegs();
          for (unsigned i = Q.interferingVRegs().size(); i; --i) {
            LiveInterval *Intf = Q.interferingVRegs()[i - 1];
            if (!Intf->isSpillable()) {
	      spillable = false;
	      break;
	    } else {
		ret[PhysReg] += Intf->weight;
	    }
          }
	  float max = std::numeric_limits<float>::max();
	if (ret[PhysReg] < -max) {
	  ret[PhysReg] = 0.0;
	}
      } else {
        break;
      }
    }
  }
  return ret;
}

unsigned RARL::pickAction(std::vector<float> state, SmallVector<unsigned, 8>& cand) {

  unsigned action = policy->pick_action(state, cand);
  return action;
}


void RARL::observe(std::vector<float> old_state, unsigned action, float reward, std::vector<float> new_state) {
  learn->observe(old_state, action, reward, new_state, past_cand);
}

// Driver for the register assignment and splitting heuristics.
// Manages iteration over the LiveIntervalUnions.
//
// This is a minimal implementation of register assignment and splitting that
// spills whenever we run out of registers.
//
// selectOrSplit can only be called once per live virtual register. We then do a
// single interference test for each register the correct class until we find an
// available register. So, the number of interference tests in the worst case is
// |vregs| * |machineregs|. And since the number of interference tests is
// minimal, there is no value in caching them outside the scope of
// selectOrSplit().
unsigned RARL::selectOrSplit(LiveInterval &VirtReg,
                                SmallVectorImpl<unsigned> &SplitVRegs,
				unsigned prev_action,
				int prev_reward,
				float &weight) {
  // Populate a list of physical register spill candidates.
  SmallVector<unsigned, 8> PhysRegSpillCands;

  // Check for an available register in this class.
  AllocationOrder Order(VirtReg.reg, *VRM, RegClassInfo, Matrix);
  while (unsigned PhysReg = Order.next()) {
    // Check for interference in PhysReg
    switch (Matrix->checkInterference(VirtReg, PhysReg)) {
    case LiveRegMatrix::IK_Free:
      // PhysReg is available, allocate it.
      //return PhysReg;
      PhysRegSpillCands.push_back(PhysReg);
      continue;

    case LiveRegMatrix::IK_VirtReg:
      // Only virtual registers in the way, we may be able to spill them.
      PhysRegSpillCands.push_back(PhysReg);
      continue;

    default:
      // RegMask or RegUnit interference.
      continue;
    }
  }
  std::vector<float> new_state = get(PhysRegSpillCands, VirtReg);
  while (!PhysRegSpillCands.empty()) {

  if(!initialState){
    observe(_state, prev_action, prev_reward, new_state);
    initialState = false;
  }


  past_cand = PhysRegSpillCands;
  unsigned physReg = pickAction(new_state, PhysRegSpillCands);
  weight = new_state[physReg];
  _state = new_state;
  if (Matrix->checkInterference(VirtReg, physReg) == LiveRegMatrix::IK_Free) {
      return physReg;
  }
  // Try to spill another interfering reg with less spill weight.
  if (spillInterferences(VirtReg, physReg, SplitVRegs)) {

    assert(!Matrix->checkInterference(VirtReg, physReg) &&
           "Interference after spill.");
    // Tell the caller to allocate to this newly freed physical register.
    return physReg;
  }
  prev_reward = calculateReward(physReg, -1.0);
  prev_action = physReg;
}

  // No other spill candidates were found, so spill the current VirtReg.
  DEBUG(dbgs() << "spilling: " << VirtReg << '\n');
  if (!VirtReg.isSpillable())
    return ~0u;
  LiveRangeEdit LRE(&VirtReg, SplitVRegs, *MF, *LIS, VRM, this, &DeadRemats);
  spiller().spill(LRE);

  // The live virtual register requesting allocation was spilled, so tell
  // the caller not to allocate anything during this round.
  return 0;
}

bool RARL::runOnMachineFunction(MachineFunction &mf) {
  DEBUG(dbgs() << "********** BASIC REGISTER ALLOCATION **********\n"
               << "********** Function: "
               << mf.getName() << '\n');

  MF = &mf;
  RegAllocRL::init(getAnalysis<VirtRegMap>(),
                     getAnalysis<LiveIntervals>(),
                     getAnalysis<LiveRegMatrix>());

  calculateSpillWeightsAndHints(*LIS, *MF, VRM,
                                getAnalysis<MachineLoopInfo>(),
                                getAnalysis<MachineBlockFrequencyInfo>());

  SpillerInstance.reset(createInlineSpiller(*this, *MF, *VRM));
  if (initialState) {
  fileToMap("go.txt", g._table);
  }

  allocatePhysRegs();
  postOptimization();

  // Diagnostic output before rewriting
  DEBUG(dbgs() << "Post alloc VirtRegMap:\n" << *VRM << "\n");
  mapToFile("go.txt", g._table);
  std::cout << curr_weight << std::endl;

  releaseMemory();
  return true;
}

FunctionPass* llvm::createRLRegisterAllocator()
{
  return new RARL();
}
