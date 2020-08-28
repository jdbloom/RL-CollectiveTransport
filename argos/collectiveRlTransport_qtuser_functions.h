#ifndef COLLECTIVE_RL_TRANSPORT_QTUSER_FUNCTIONS_H
#define COLLECTIVE_RL_TRANSPORT_QTUSER_FUNCTIONS_H

#include <buzz/argos/buzz_qt.h>

using namespace argos;

class CCollectiveRLTransport;

class CCollectiveRLTransportQTUserFunctions : public CBuzzQT {

public:
  CCollectiveRLTransportQTUserFunctions();

  virtual ~CCollectiveRLTransportQTUserFunctions(){}

  virtual void DrawInWorld();

private:

  CCollectiveRLTransport& m_cCRLLF;

};

#endif
