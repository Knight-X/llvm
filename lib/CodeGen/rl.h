#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <iostream>
#include <stdlib.h>     /* srand, rand */
#include <time.h>
#include <utility>
#include <map>
#include <vector>



class RandomPolicy {
  public:
    int pick_action();
};


class QTable {
  public:
    float get(std::vector<float> state, int action);
    void set(std::vector<float> state, int action, float value);
    std::pair<int, float> best(std::vector<float> state);
  private:
    std::map<std::pair<std::vector<float>, int>, float> _table;
};

class GreedyQ {
  public:
  GreedyQ() {}
  GreedyQ(QTable* q) {
    _q = q;
  }
  int pick_action(std::vector<float> state);

  private:
    QTable* _q;
};

class EpsilonPolicy {
	
  public:
  EpsilonPolicy() {}
  EpsilonPolicy(GreedyQ* policy_a, RandomPolicy* policy_b, float epsilon){
    _greedy = policy_a;
    _random = policy_b;
    _epsilon = epsilon;
  }
  int pick_action(std::vector<float> state);

  private:
    GreedyQ* _greedy;
    RandomPolicy* _random;
    float _epsilon;
};

class QLearner {
  public:
    QLearner() {
    }
    QLearner(QTable* q, float learning_rate, float discount_rate) {
      _q = q;
      _alpha = learning_rate;
      _gamma = discount_rate;
    }

    void observe(std::vector<float>, int action, float reward, std::vector<float>);
  private:
   float _alpha;
   float _gamma; 
   QTable* _q;
};


/*class MachinePlayer {

  public:
    MachinePlayer() {}
  MachinePlayer(EpsilonPolicy policy, QLearner learner) {
    _policy = policy;
    _learner = learner;
  }
  void interact(Simulation&);
  private:
    EpsilonPolicy _policy;
    QLearner _learner;
};

class BasicBlock {
  public:
  BasicBlock(MachinePlayer driver) {
      _sim = Simulation();
      _wins = 0;
      _losses = 0;
      _was_in_terminal_state = false;
      _driver = driver;
    }
 
  void step(bool);
  void _draw();
  private:
    int _wins;
    int _losses;
    bool _was_in_terminal_state;
    MachinePlayer _driver;
    Simulation _sim;
};*/

