#include "cmPrediction_qtuser_functions.h"
#include "cmPrediction.h"


CCMPredictionQTUserFunctions::CCMPredictionQTUserFunctions() :
  m_cCRLLF(dynamic_cast<CCMPrediction&>(CSimulator::GetInstance().GetLoopFunctions())){
  }

/****************************************/
/****************************************/

void CCMPredictionQTUserFunctions::DrawInWorld(){
  /*Draw the threshold on the floor*/
  CBuzzQT::DrawInWorld();
  DrawCircle(CVector3(0.0, 0.0, 0.05f),
             CQuaternion(),
             0.0,
             CColor::BLACK);
}

REGISTER_QTOPENGL_USER_FUNCTIONS(CCMPredictionQTUserFunctions, "cm_prediction_qtuser_functions")
