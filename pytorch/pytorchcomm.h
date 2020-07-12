#ifndef PYTORCHCOMM_H
#define PYTORCHCOMM_H

#include "zmq.hpp"

#include <vector>
#include <string>

class CPyTorchComm {
   
public:
   
   CPyTorchComm(unsigned int un_port,
                unsigned int un_num_agents,
                unsigned int un_num_obs,
                unsigned int un_num_actions);

   void Init();

   /**
    * Sends observations to PyTorch.
    * @param vec_obs The observations
    * @return 0 if successful, -1 on error.
    */
   void SendObservations(const std::vector<float>& vec_obs);

   /**
    * Returns the actions selected by PyTorch.
    */
   std::vector<float> GetActions();

   /**
    * Ends communication with PyTorch.
    * @return ???
    */
   void Disconnect();

private:

   /**
    * Sends a buffer to PyTorch.
    * @param
    * @return ???
    */
   void Send(const std::vector<float>& vec_send_buffer);
   
   void Send(const std::vector<float>& vec_send_buffer,
             std::vector<float>& vec_response_buffer);

private:

   /** ZeroMQ context */
   zmq::context_t m_ptContext;
   zmq::socket_t m_tSocket;
   std::string m_strURL;
   unsigned int m_unNumAgents;
   unsigned int m_unNumObs;
   unsigned int m_unNumActions;
   
};

#endif
