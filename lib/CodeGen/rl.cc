#include "rl.h"
#include <limits>

/*void BasicBlock::step(bool ter) {
   if (ter) {
     _sim.terminal_state = true;  
   }

   _driver.interact(_sim);
   if (_sim.terminal_state){
     if (_sim._score < 0) {
       _losses += 1;
     } else {
       _wins += 1;
     }
     _draw();
    _sim.terminal_state = false;
   }
   _was_in_terminal_state = _sim.terminal_state;
}

void BasicBlock::_draw() {
   printf("the winner score is %f", _sim.prev_weight);
} */


int RandomPolicy::pick_action(SmallVector<unsigned, 8>& cand) {
    int i = rand() % cand.size();
    int ret = cand[i];
    cand.erase(cand.begin() + i);
    return ret;
}

int GreedyQ::pick_action(std::vector<float> state, SmallVector<unsigned, 8>& cand) {
    return _q->best(state, cand).first;
}

int EpsilonPolicy::pick_action(std::vector<float> state, SmallVector<unsigned, 8>& cand) {
	float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    if (r < _epsilon)
      return _random->pick_action(cand);
    else 
      return _greedy->pick_action(state, cand);
}

float QTable::get(std::vector<float> state, int action) {
   std::pair<std::vector<float>, int> n_state = std::make_pair(state, action);
    return _table[n_state];
}


void QTable::set(std::vector<float> state, int action, float value) {
  std::pair<std::vector<float>, int> n_state = std::make_pair(state, action);
  _table[n_state] = value;
}


std::pair<int, float>  QTable::best(std::vector<float> state, SmallVector<unsigned, 8>& cand) {
  float best_value = -std::numeric_limits<float>::max();
  int best_action = 0;
  int index = -1;
  for (unsigned i = 0; i < cand.size(); i++) {
    int action = cand[i];
    float value = get(state, action);
      if (value > best_value) {
        best_action = action;
        best_value = value;
	index = i;
      }
  }
  std::pair<int, float> n_state = std::make_pair(best_action, best_value);
  cand.erase(cand.begin() + index);
  return n_state; 
}


void QLearner::observe(std::vector<float> old_state, int action, float reward, std::vector<float> new_state, SmallVector<unsigned, 8>& cand) {
  float prev = _q->get(old_state, action);
  _q->set(old_state, action, prev + _alpha * (
        reward + _gamma * _q->best(new_state, cand).second - prev));
}

/*void MachinePlayer::interact(Simulation& sim) {
  if (sim.terminal_state) {
    sim.terminal();
    sim.reset();
  } else {
    std::vector<float> old_state = sim._state;
    int action = _policy.pick_action(sim._state);
    printf("action is %d\r\n", action);
    float reward = sim.allocate(action);
    _learner.observe(old_state, action, reward, sim._state);
  } 
}*/
