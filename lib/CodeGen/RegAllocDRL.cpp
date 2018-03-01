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
#include <unistd.h>
#include <queue>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <stdio.h>
#include "RegAllocBase.h"
#include "Spiller.h"
#include <iostream>
#include <fstream>
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

using namespace llvm;
typedef std::map<unsigned, std::set<std::pair<int, int>>> commap;

#define DEBUG_TYPE "regallocdl"

static RegisterRegAlloc drlRegAlloc("drl", "drl register allocator",
                                      createDRLRegisterAllocator);

static float weight = 0.0;
namespace {
  struct CompSpillWeight {
    bool operator()(LiveInterval *A, LiveInterval *B) const {
      return A->weight < B->weight;
    }
  };
}
 
enum VirgState {
	Free = 0,
	Virt,
	Self,
	Failed
};

bool mapToFilep(const std::string &filename, commap &store, std::map<int, float>& reward)     //Write Map
{
    std::ofstream ofile;
    ofile.open(filename.c_str());
    if(!ofile)
    {
        return false;           //file does not exist and cannot be created.
    }
    ofile << 3333;
    for (auto iter : store[3333]) {
      ofile << "&" << iter.first << "&" << iter.second << "\n";
    }
    if (reward.size() > 0) {
      ofile << "reward\n";
      for (std::map<int, float>::iterator iter = reward.begin(); iter != reward.end(); iter++) {
        ofile << iter->first << "&" << iter->second << "&";
      }
      ofile << "\n";
    }

    for (commap::iterator it=store.begin(); it!=store.end(); ++it) {
	if (it->first != 3333) {
          ofile << it->first;
	  for(auto iter : it->second) {
	    ofile << "&" << iter.first << "&" << iter.second;
	  }
	  ofile<< "\n";
	}
    }	
    return true;
}

namespace {
/// RABasic provides a minimal implementation of the basic register allocation
/// algorithm. It prioritizes live virtual registers by spill weight and spills
/// whenever a register is unavailable. This is not practical in production but
/// provides a useful baseline both for measuring other allocators and comparing
/// the speed of the basic algorithm against other styles of allocators.
class RADrl : public MachineFunctionPass,
                public RegAllocBase,
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

public:
  RADrl();

  /// Return the pass name.
  StringRef getPassName() const override { return "Basic Register Allocator"; }

  /// RABasic analysis usage.
  void getAnalysisUsage(AnalysisUsage &AU) const override;

  void printPhysic(LiveRange& vreg, commap& state);

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
                         SmallVectorImpl<unsigned> &SplitVRegs) override;
bool doFinalization(Module &M) override {
  DEBUG(llvm::dbgs() << "terminal" << "\n");
  int sockfd = 0;
  sockfd = socket(AF_INET, SOCK_STREAM, 0);  
  struct sockaddr_in info;
  bzero(&info, sizeof(info));
  info.sin_family = PF_INET;

  info.sin_addr.s_addr = inet_addr("127.0.0.1");
  info.sin_port = htons(1992);

  connect(sockfd, (struct sockaddr *)&info, sizeof(info));	

  const char g = 'e';
  int n = send(sockfd, &g, 1, 0);
  close(sockfd);
  return true;
}

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
  bool calculateReward(std::map<int, float>& reward, SmallVectorImpl<unsigned>& cands, SmallVectorImpl<unsigned>& vrcand, commap& state);
  bool calculateSpillWeight(LiveInterval &Virt, unsigned Phys, float &weight);
  unsigned pickAction(std::map<int, float>& reward, commap& state);
  int checkGroup(unsigned reg, SmallVectorImpl<unsigned>& cands, SmallVectorImpl<unsigned>& vrcand, std::map<int, float>& reward);

  static char ID;
};

char RADrl::ID = 0;

} // end anonymous namespace

char &llvm::RADrlID = RADrl::ID;

INITIALIZE_PASS_BEGIN(RADrl, "regallocdrl", "Drl Register Allocator",
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
INITIALIZE_PASS_END(RADrl, "regallocdrl", "Drl Register Allocator", false,
                    false)

bool RADrl::LRE_CanEraseVirtReg(unsigned VirtReg) {
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
static int iteration = 1;
void RADrl::printPhysic(LiveRange& vreg, commap& state) {
    DEBUG(llvm::dbgs() << "new state: " << std::to_string(iteration) << "\n");
    LiveIntervalUnion *Q = Matrix->getLiveUnions();
    const TargetRegisterInfo *TRI = &VRM->getTargetRegInfo();
    for (unsigned i = 0; i < TRI->getNumRegClasses(); i++) {
      const TargetRegisterClass* g = TRI->getRegClass(i);
      ArrayRef<MCPhysReg> group = RegClassInfo.getOrder(g);
      for (unsigned index = 0; index < group.size(); index++){
	unsigned PhysReg = group[index];
        for (MCRegUnitIterator Units(PhysReg, TRI); Units.isValid(); ++Units) {
	  IntervalMap<SlotIndex, LiveInterval*>::const_iterator LiveUnionI;
	  LiveUnionI.setMap(Q[*Units].getMap());
	  LiveUnionI.goToBegin();
	  while (LiveUnionI.valid()) {
		  LiveRange::const_iterator LRI = vreg.begin();
	    if (LiveUnionI.start() > LRI->start && LiveUnionI.start().getint() < LRI->start.getint() + 246)
	    state[PhysReg].insert(std::pair<int, int>(LiveUnionI.start().getint(), LiveUnionI.stop().getint()));
          //std::cout << "start: " << LiveUnionI.start().getint() << "end: " << LiveUnionI.stop().getint()<< std::endl;
	  LiveUnionI++;
	  }
        }
      }
    }
    LiveRange::const_iterator LRI = vreg.begin();
    state[3333].insert(std::pair<int, int>(LRI->start.getint(), LRI->end.getint()));
    DEBUG(llvm::dbgs() << "new state: " << std::to_string(iteration) << "finish\n");
    iteration++;
    /*for (std::map<unsigned, std::set<std::pair<int, int>>>::iterator it=store.begin(); it!=store.end(); ++it) {
        std::cout << "phys reg: " << it->first << std::endl;
	for(auto iter : it->second) {
	  std::cout << "start: " << iter.first << "stop: " << iter.second << " ";
	}
	std::cout << std::endl;
    }	
    std::cout << std::endl;*/

}
void RADrl::LRE_WillShrinkVirtReg(unsigned VirtReg) {
  if (!VRM->hasPhys(VirtReg))
    return;

  // Register is assigned, put it back on the queue for reassignment.
  LiveInterval &LI = LIS->getInterval(VirtReg);
  Matrix->unassign(LI);
  enqueue(&LI);
}

RADrl::RADrl(): MachineFunctionPass(ID) {
}

void RADrl::getAnalysisUsage(AnalysisUsage &AU) const {
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

void RADrl::releaseMemory() {
  SpillerInstance.reset();
}


bool RADrl::calculateSpillWeight(LiveInterval &Virt, unsigned Phys, float &w) {
  SmallVector<LiveInterval*, 8> Intfs;

  // Collect interferences assigned to any alias of the physical register.
  for (MCRegUnitIterator Units(Phys, TRI); Units.isValid(); ++Units) {
    LiveIntervalUnion::Query &Q = Matrix->query(Virt, *Units);
    Q.collectInterferingVRegs();
    for (unsigned i = Q.interferingVRegs().size(); i; --i) {
      LiveInterval *Intf = Q.interferingVRegs()[i - 1];
      if (!Intf->isSpillable()) {
	return false;
      }
      w += Intf->weight;
    }
  }

  return true;
}

unsigned RADrl::pickAction(std::map<int, float>& reward, commap& state) {
  std::string file = "state.txt";
  mapToFilep(file, state, reward);
  std::string filename = "statedone.txt";
  std::ofstream ofile;
  ofile.open(filename.c_str());
  if(!ofile)
  {
      return false; 
  }
  int sockfd = 0;
  sockfd = socket(AF_INET, SOCK_STREAM, 0);  
  struct sockaddr_in info;
  bzero(&info, sizeof(info));
  info.sin_family = PF_INET;

  info.sin_addr.s_addr = inet_addr("127.0.0.1");
  info.sin_port = htons(1992);

  connect(sockfd, (struct sockaddr *)&info, sizeof(info));	

  char g[11] = {};
  char receive[3] = {};
  sprintf(g, "%d", iteration);
  int n = send(sockfd, g, 1, 0);
  recv(sockfd, receive, sizeof(receive), 0);
  close(sockfd);
  int j;
  sscanf(receive, "%d", &j);
  unsigned action = (unsigned)j;
  return action;
}
int RADrl::checkGroup(unsigned reg, SmallVectorImpl<unsigned>& cands, SmallVectorImpl<unsigned>& vrcand, std::map<int, float>& reward) {
DEBUG(dbgs() << "phys\n"; for (unsigned i = 0; i < cands.size(); i++) { llvm::dbgs() << cands[i] << " "; } dbgs() << "\n";);
DEBUG(dbgs() << "virts\n"; for (unsigned i = 0; i < vrcand.size(); i++) { llvm::dbgs() << vrcand[i] << " "; } dbgs() << "\n";);
 for (unsigned candsi = 0; candsi < cands.size(); candsi++) {
	 if (cands[candsi] == reg) {
		 std::map<int, float>::iterator it = reward.find(reg);
		 if (it != reward.end()) {
		   reward.erase(it);
		 }
           DEBUG(llvm::dbgs() << "choose phys" << reg << "\n");
	   return Free;
	 }
 }

 for (unsigned vrcandi = 0; vrcandi < vrcand.size(); vrcandi++) {
   if (vrcand[vrcandi] == reg) {
	  std::map<int, float>::iterator it = reward.find(reg);
	  if (it != reward.end()) {
	    reward.erase(it);
	   }
           DEBUG(llvm::dbgs() << "choose virts" << reg << "\n");
	   vrcand.erase(vrcand.begin() + vrcandi);
	   return Virt;
   }

 }
   DEBUG(llvm::dbgs() << "choose nothing" << reg << "\n");
   return 3;
}
bool RADrl::calculateReward(std::map<int, float>& reward, SmallVectorImpl<unsigned>& cands, SmallVectorImpl<unsigned>& vrcand, commap& state) {
  DEBUG(llvm::dbgs() << "c++: reward start \n");
  struct cmp {
        bool operator()(const std::pair<unsigned, float> &a, const std::pair<unsigned, float> &b) {
            return a.second < b.second;
        };
  };
  std::priority_queue<std::pair<unsigned, float>, std::vector<std::pair<unsigned, float>>, cmp> p(reward.begin(), reward.end());
DEBUG(dbgs() << "phys\n"; for (unsigned i = 0; i < cands.size(); i++) { llvm::dbgs() << cands[i] << " "; } dbgs() << "\n";);
DEBUG(dbgs() << "virts\n"; for (unsigned i = 0; i < vrcand.size(); i++) { llvm::dbgs() << vrcand[i] << " "; } dbgs() << "\n";);
  for (unsigned i = 0; i < vrcand.size(); i++) {
    std::pair<unsigned, float> top = p.top();
    reward[top.first] =  vrcand.size() - i;
    p.pop();
  }
  for (unsigned i = 0; i < cands.size(); i++) {
    reward[cands[i]] = cands.size() - i;
  }
  DEBUG(dbgs() << "rewards\n"; for (std::map<int, float>::iterator it = reward.begin(); it != reward.end(); it++) { llvm::dbgs() << "first: " << it->first << " second: " << it->second << " "; } dbgs() << "\n";);

  //std::string file = "go" + std::to_string(iteration) + ".txt";
  DEBUG(llvm::dbgs() << "c++: reward end \n");
  return true;
}
// Spill or split all live virtual registers currently unified under PhysReg
// that interfere with VirtReg. The newly spilled or split live intervals are
// returned by appending them to SplitVRegs.
bool RADrl::spillInterferences(LiveInterval &VirtReg, unsigned PhysReg,
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
      if (!Intf->isSpillable() || Intf->weight > VirtReg.weight)
        return false;
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
    weight += Spill.weight;
    spiller().spill(LRE);
  }
  return true;
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
unsigned RADrl::selectOrSplit(LiveInterval &VirtReg,
                                SmallVectorImpl<unsigned> &SplitVRegs) {
  // Populate a list of physical register spill candidates.
  SmallVector<unsigned, 8> VRPhysRegSpillCands;
  SmallVector<unsigned, 8> PhysRegSpillCands;
  std::map<int, float> reward;

  // Check for an available register in this class.
  AllocationOrder Order(VirtReg.reg, *VRM, RegClassInfo, Matrix);
    std::map<unsigned, std::set<std::pair<int, int>>> state;
  printPhysic(VirtReg, state);
  while (unsigned PhysReg = Order.next()) {
    float new_weight = 0.0;
    // Check for interference in PhysReg
    switch (Matrix->checkInterference(VirtReg, PhysReg)) {
    case LiveRegMatrix::IK_Free:
      // PhysReg is available, allocate it.
      PhysRegSpillCands.push_back(PhysReg);
      continue;

    case LiveRegMatrix::IK_VirtReg:
      // Only virtual registers in the way, we may be able to spill them.
      if (calculateSpillWeight(VirtReg, PhysReg, new_weight)) {
        VRPhysRegSpillCands.push_back(PhysReg);
        reward[PhysReg] = new_weight;
      }
      continue;

    default:
      // RegMask or RegUnit interference.
      continue;
    }
  }
  if (VirtReg.isSpillable()) {

    DEBUG(llvm::dbgs() << "virt self is spillable \n");
    VRPhysRegSpillCands.push_back(0);
    reward[0] = VirtReg.weight;
  }
  calculateReward(reward, PhysRegSpillCands, VRPhysRegSpillCands, state);
    DEBUG(llvm::dbgs() << "state done \n");
  // Try to spill another interfering reg with less spill weight.
  while (unsigned Reg = pickAction(reward, state)) {
    DEBUG(llvm::dbgs() << "pick finish " << Reg << "\n");
    switch (checkGroup(Reg, PhysRegSpillCands, VRPhysRegSpillCands, reward)) {
    case Free:
      DEBUG(llvm::dbgs() << "check phys finish " << Reg << "\n");
      return Reg;
    case Virt:
      DEBUG(llvm::dbgs() << "check virtual finish " << Reg << "\n");
      if (!spillInterferences(VirtReg, Reg, SplitVRegs))
        continue;

      assert(!Matrix->checkInterference(VirtReg, Reg) &&
           "Interference after spill.");
      // Tell the caller to allocate to this newly freed physical register.
      return Reg;

    case Failed: {
      while (true) {
      DEBUG(llvm::dbgs() << "candidate is wrong, but target is" << Reg << "\n");
      DEBUG(dbgs() << "phys\n"; for (unsigned i = 0; i < PhysRegSpillCands.size(); i++) { llvm::dbgs() << PhysRegSpillCands[i] << " "; } dbgs() << "\n";);
      DEBUG(dbgs() << "virts\n"; for (unsigned i = 0; i < VRPhysRegSpillCands.size(); i++) { llvm::dbgs() << VRPhysRegSpillCands[i] << " "; } dbgs() << "\n";);
      }
      continue;
    }

    }
  }
      
      DEBUG(dbgs() << "spilling self: " << VirtReg << '\n');
      if (!VirtReg.isSpillable())
        return ~0u;
      std::map<int, float>::iterator it = reward.find(0);
      if (it != reward.end()) {
	reward.erase(it);
      }
      LiveRangeEdit LRE(&VirtReg, SplitVRegs, *MF, *LIS, VRM, this, &DeadRemats);
      spiller().spill(LRE);
      return 0;
      // No other spill candidates were found, so spill the current VirtReg.
}

bool RADrl::runOnMachineFunction(MachineFunction &mf) {
  DEBUG(dbgs() << "********** BASIC REGISTER ALLOCATION **********\n"
               << "********** Function: "
               << mf.getName() << '\n');

  std::cout << "new function" << std::endl;
  MF = &mf;
  RegAllocBase::init(getAnalysis<VirtRegMap>(),
                     getAnalysis<LiveIntervals>(),
                     getAnalysis<LiveRegMatrix>());

  calculateSpillWeightsAndHints(*LIS, *MF, VRM,
                                getAnalysis<MachineLoopInfo>(),
                                getAnalysis<MachineBlockFrequencyInfo>());

  SpillerInstance.reset(createInlineSpiller(*this, *MF, *VRM));

  allocatePhysRegs();
  postOptimization();

  // Diagnostic output before rewriting
  //DEBUG(dbgs() << "Post alloc VirtRegMap:\n" << *VRM << "\n");

  releaseMemory();
  std::cout << "function end" << std::to_string(iteration) << std::endl;
  return true;
}

FunctionPass* llvm::createDRLRegisterAllocator()
{
  return new RADrl();
}
