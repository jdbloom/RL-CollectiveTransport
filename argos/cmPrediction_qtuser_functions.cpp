#include "cmPrediction_qtuser_functions.h"
#include "cmPrediction.h"
#include <argos3/plugins/simulator/physics_engines/dynamics2d/dynamics2d_multi_body_object_model.h>


CCMPredictionQTUserFunctions::CCMPredictionQTUserFunctions() :
  m_cCRLLF(dynamic_cast<CCMPrediction&>(CSimulator::GetInstance().GetLoopFunctions())){
    RegisterUserFunction<CCMPredictionQTUserFunctions,CKheperaIVEntity>(&CCMPredictionQTUserFunctions::Draw);
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

void CCMPredictionQTUserFunctions::Draw(CKheperaIVEntity& cKheperaIV){
  CDynamics2DKheperaIVModel& cModel = dynamic_cast<CDynamics2DKheperaIVModel&>(cKheperaIV.GetEmbodiedEntity().GetPhysicsModel(0));
  // cpBody* ptBaseBody = cModel.GetActualBaseBody();
  // cpBody* ptGripperBody = cModel.GetActualGripperBody();
  CDynamics2DMultiBodyObjectModel::SBody& cBody1 = cModel.GetBody(0);
  cpVect pos1 = cBody1.OffsetPos;
  cpFloat angle1 = cBody1.OffsetOrient;
  CQuaternion cCylinderOrient = CQuaternion();
  cCylinderOrient.FromEulerAngles(CRadians::ZERO,CRadians::ZERO,CRadians(angle1));
  CVector3 cCylinderPos = CVector3(pos1.x,pos1.y, 0.06);
  CDynamics2DMultiBodyObjectModel::SBody& cBody = cModel.GetBody(1);
  cpVect pos = cBody.OffsetPos;
  cpFloat angle = cBody.OffsetOrient;
  CQuaternion gripBodyQuat = CQuaternion();
  gripBodyQuat.FromEulerAngles(CRadians::ZERO,CRadians::ZERO,CRadians(angle));
  CVector3 posvec = CVector3(pos.x,pos.y, 0.09);

  // CQuaternion baseBodyQuat = CQuaternion();
  // baseBodyQuat.FromEulerAngles(CRadians::ZERO,CRadians::ZERO,CRadians(ptBaseBody->a));
  // printf("%f", ptBaseBody->a);
  // CQuaternion gripBodyQuat = CQuaternion();
  // gripBodyQuat.FromEulerAngles(CRadians::ZERO,CRadians::ZERO,CRadians(ptBaseBody->a));
  // printf("%f, %f top plate %f, %f\n", ptBaseBody->p.x, ptBaseBody->p.y, ptGripperBody->p.x, ptGripperBody->p.y);
  // printf("%f, %f top plate %f, %f\n", cCylinderPos.GetX(), cCylinderPos.GetY(), pos.x, pos.y);
  DrawCircle(cCylinderPos, cCylinderOrient, KHEPERAIV_GRIPPER_RING_RADIUS, CColor::BLUE, false);
  DrawCircle(posvec, gripBodyQuat, KHEPERAIV_GRIPPER_RING_RADIUS, CColor::GREEN, false);
}

REGISTER_QTOPENGL_USER_FUNCTIONS(CCMPredictionQTUserFunctions, "cmPrediction_qtuser_functions")
