#include "pytorchcomm.h"

#include <cstring>
#include <iostream>

/****************************************/
/****************************************/

static const float SEND_INIT_SERVER     = 0.0;
static const float SEND_OBSERVATION     = 1.0;
static const float REQUEST_ACTION       = 2.0;
static const float REQUEST_CLOSE_SERVER = 3.0;

/****************************************/
/****************************************/

CPyTorchComm::CPyTorchComm(unsigned int un_port,
                           unsigned int un_num_agents,
                           unsigned int un_num_obs,
                           unsigned int un_num_actions) :
   m_ptContext(zmq::context_t(1)), // TODO WHAT IS 1???
   m_tSocket(m_ptContext, ZMQ_REQ),
   m_unNumAgents(un_num_agents),
   m_unNumObs(un_num_obs),
   m_unNumActions(un_num_actions) {
   std::ostringstream ossURL;
   ossURL << "tcp://localhost:" << un_port;
   m_strURL = ossURL.str();
}

/****************************************/
/****************************************/

void CPyTorchComm::Init() {
   /* Connect to socket */
   m_tSocket.connect(m_strURL);
   /* Send a message to PyTorch to initialize the connection */
   std::vector<float> vecMsg(4);
   vecMsg[0] = SEND_INIT_SERVER;
   vecMsg[1] = m_unNumAgents;
   vecMsg[2] = m_unNumObs;
   vecMsg[3] = m_unNumActions;
   Send(vecMsg);
}

/****************************************/
/****************************************/

void CPyTorchComm::SendObservations(const std::vector<float>& vec_obs) {
   std::vector<float> vecMsg = vec_obs;
   vecMsg.insert(vecMsg.begin(), SEND_OBSERVATION);
   Send(vecMsg);
}

/****************************************/
/****************************************/

std::vector<float> CPyTorchComm::GetActions() {
   std::vector<float> vecMsg(1, REQUEST_ACTION);
   std::vector<float> vecResp(m_unNumAgents * m_unNumActions);
   Send(vecMsg, vecResp);
   return vecResp;
}

/****************************************/
/****************************************/

void CPyTorchComm::Disconnect() {
   std::vector<float> vecMsg(1, REQUEST_CLOSE_SERVER);
   Send(vecMsg);
}

/****************************************/
/****************************************/

void CPyTorchComm::Send(const std::vector<float>& vec_send_buffer) {
  /* Sends sendBuffer to the server, writes response to responseBuffer.
   * If responseBuffer is NULL, no response will be read from the server.
   * Note that the size of the buffers is in FLOATS NOT BYTES
   */
  zmq::message_t tReq(vec_send_buffer.size() * sizeof(float));
  memcpy(tReq.data(),
         &vec_send_buffer[0],
         vec_send_buffer.size() * sizeof(float));
  m_tSocket.send(tReq, zmq::send_flags::none);
  zmq::message_t tReply;
  m_tSocket.recv(tReply);
}

/****************************************/
/****************************************/

void CPyTorchComm::Send(const std::vector<float>& vec_send_buffer,
                           std::vector<float>& vec_response_buffer) {
  /* Sends sendBuffer to the server, writes response to responseBuffer.
   * If responseBuffer is NULL, no response will be read from the server.
   * Note that the size of the buffers is in FLOATS NOT BYTES
   */
  zmq::message_t tRequest(vec_send_buffer.size() * sizeof(float));
  memcpy(tRequest.data(),
         &vec_send_buffer[0],
         vec_send_buffer.size() * sizeof(float));
  m_tSocket.send(tRequest, zmq::send_flags::none);
  zmq::message_t tReply;
  m_tSocket.recv(tReply);
  memcpy(&vec_response_buffer[0],
         tReply.data(),
         vec_response_buffer.size() * sizeof(float));
}

/****************************************/
/****************************************/
