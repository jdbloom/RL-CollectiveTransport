#ifndef ModelServerClient_hpp
#define ModelServerClient_hpp
#include <zmq.hpp>

class ModelServerClient {
private:
  zmq::context_t _context;
  int _envPort;
  int _modelPort;
  int _numAgents;
  int _sizeObs;
  int _sizeAction;
  int SendBufferToServer(float *sendBuffer, int buff1Size, float *responseBuffer, int buff2Size);
  int InitServer();
public:
  ModelServerClient(int envPort, int modelPort, int numAgents, int sizeObs, int sizeAction);
  int SendAgentObservations(float *observation);
  float* GetAgentActions();
  int CloseServer();
};

#endif
