#include "collectiveRlTransport_qtuser_functions.h"
#include "collectiveRlTransport.h"


CCollectiveRLTransportQTUserFunctions::CCollectiveRLTransportQTUserFunctions() :
  m_cCRLLF(dynamic_cast<CCollectiveRLTransport&>(CSimulator::GetInstance().GetLoopFunctions())){
  }

/****************************************/
/****************************************/

void CCollectiveRLTransportQTUserFunctions::DrawInWorld(){
  /*Draw the threshold on the floor*/
  CBuzzQT::DrawInWorld();
  DrawCircle(CVector3(m_cCRLLF.GetGoal().GetX(), m_cCRLLF.GetGoal().GetY(), 0.05f),
             CQuaternion(),
             m_cCRLLF.GetGoal().GetZ(),
             CColor::BLACK);
}

REGISTER_QTOPENGL_USER_FUNCTIONS(CCollectiveRLTransportQTUserFunctions, "collective_rl_transport_qtuser_functions")
