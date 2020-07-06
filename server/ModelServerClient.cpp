#include "ModelServerClient.hpp"
#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include <string.h>
#include <assert.h>
#include <iostream>

#define SEND_OBSERVATION INFINITY
#define REQUEST_ACTION -INFINITY
#define REQUEST_CLOSE_SERVER NAN
#define SEND_INIT_SERVER -0.0

ModelServerClient::ModelServerClient(int envPort, int modelPort, int numAgents, int sizeObs, int sizeAction) {
  _envPort = envPort;
  _modelPort = modelPort;
  _numAgents = numAgents;
  _sizeObs = sizeObs;
  _sizeAction = sizeAction;
  _context = zmq::context_t (1);
  InitServer();
}

/**
 * Takes the agentId, a pointer to an array of floats representing
 * the observations, and the number of observations.
 * Sends the observation to the python model.
 * Returns: 0 if successful, -1 on error.
 **/
int ModelServerClient::SendAgentObservations(float *observation) {
  float data[((_sizeObs)*_numAgents)+1];
  data[0] = SEND_OBSERVATION;
  for(int i = 1; i <= (_sizeObs) * _numAgents; i++) {
    data[i] = observation[i-1];
  }

  SendBufferToServer(data, ((_sizeObs)*_numAgents) +1, NULL, 0);
  return 0;
}

float* ModelServerClient::GetAgentActions() {
  float *msgBuffer = new float[2];
  msgBuffer[0] = REQUEST_ACTION;
  float *responseBuffer = new float[1024];
  SendBufferToServer(msgBuffer, 1, responseBuffer, 1024);
  return responseBuffer;
}

int ModelServerClient::CloseServer() {
  float data[1];
  data[0] = REQUEST_CLOSE_SERVER;
  SendBufferToServer(data, 1, NULL, 0);
  return 0;
}

int ModelServerClient::InitServer() {
  float msgBuffer[4];
  msgBuffer[0] = SEND_INIT_SERVER;
  msgBuffer[1] = _numAgents;
  msgBuffer[2] = _sizeObs;
  msgBuffer[3] = _sizeAction;
  SendBufferToServer(msgBuffer, 4, NULL, 0);
  return 0;
}

int ModelServerClient::SendBufferToServer(float *sendBuffer, int buff1Size, float *responseBuffer, int buff2Size) {
  /* Sends sendBuffer to the server, writes response to responseBuffer.
   * If responseBuffer is NULL, no response will be read from the server.
   * Note that the size of the buffers is in FLOATS NOT BYTES
   */
  zmq::socket_t socket (_context, ZMQ_REQ);
  socket.connect("tcp://localhost:5555");
  zmq::message_t request(buff1Size*sizeof(float));
  memcpy(request.data(), sendBuffer, buff1Size*sizeof(float));
  socket.send(request);
  zmq::message_t reply;
  socket.recv(&reply);
  if(responseBuffer) {
    memcpy(responseBuffer, reply.data(), buff2Size*sizeof(float));
  }
  return 0;
}
