#include "pytorchcomm.h"

static const unsigned int PORT        = 5555;
static const unsigned int NUM_AGENTS  = 4;
static const unsigned int NUM_OBS     = 3;
static const unsigned int NUM_ACTIONS = 2;


int main(int argc, char* argv[]) {
   CPyTorchComm cPTC(PORT, NUM_AGENTS, NUM_OBS, NUM_ACTIONS);
   if(!cPTC.Init()) {
      abort();
   }
   std::vector<float> vecObs;
   if(!cPTC.SendObservations(vecObs)) {
      abort();
   }
   std::vector<float> vecActions = cPTC.GetActions();
   if(!vecActions.empty()) {
      abort();
   }
}
