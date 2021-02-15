#include <argos3/core/utility/networking/tcp_socket.h>

using namespace argos;

CByteArray& appendFloatIEEE754(CByteArray& b, float f){
  b << (*reinterpret_cast<UInt32*>(&f));
  return b;
}
CByteArray& getFloatIEEE754(CByteArray& b, float& f){
  UInt32 buffer;
  b >> buffer;
  f = *reinterpret_cast<float*>(&buffer);
  return b;
}


int main() {
   try {
      /*
      Connect
      */
      CTCPSocket s;
      std::cout << "[INFO] Connecting to localhost: 65432" << std::endl;
      s.Connect("localhost", 65432);
      std::cout << "[INFO] Connected to server" << std::endl;
      /*
      Send Params
        Num Robots      UInt32
        Num Obs         UInt32
        Num Actions     UInt32
        Num Stats       UInt32
        Alphabet Size   UInt32
      */
      UInt32 numRobots = 4;
      UInt32 numObs = 31;
      UInt32 numActions = 2;
      UInt32 numStats = 3;
      UInt32 alphabetSize = 1;

      CByteArray params;
      params << numRobots << numObs << numActions << numStats << alphabetSize;
      std::cout<<"[SOCKET] Sending Params: " << params <<std::endl;
      s.SendMsg(params);
      /*
      Get Ack
      */
      CByteArray ack;
      s.RecvMsg(ack);
      if(ack[0] == 111 && ack[1] == 107){
        std::cout << "[SOCKET] Received Ack from server"<<std::endl;
      }
      /*
      Send Multipart
        Episode State       vec<uint16_t>
        Env Observations    vec<float>
        Failures            vec<int>
        Rewards             vec<float>
        Stats               vec<float>
      */
      std::vector<uint16_t> eState;
      std::vector<float> envObs;
      std::vector<int> failures;
      std::vector<float> rewards;
      std::vector<float> stats;

      CByteArray bState;
      CByteArray bEnvObs;
      CByteArray bFailures;
      CByteArray bRewards;
      CByteArray bStats;

      eState.push_back(0);
      eState.push_back(1);
      eState.push_back(2);
      for(size_t j = 0; j < numRobots; ++j){
        for(size_t i = 0; i < numObs; ++i){
          envObs.push_back(3.7);
        }
      }

      for(size_t i = 0; i < numRobots; ++i){
        failures.push_back(0);
        rewards.push_back(2.25);
      }
      for(size_t i = 0; i < numRobots; ++i){
        stats.push_back(5.5);
        stats.push_back(7.5);
        stats.push_back(3.5);
      }

      bState << eState[0] << eState[1] << eState[2];
      for(size_t i = 0; i < envObs.size(); ++i){
        appendFloatIEEE754(bEnvObs, envObs[i]);
        std::cout<<"[DEBUG] Writing "<<envObs[i]<< " "<< typeid(envObs[i]).name()<<std::endl;
        std::cout<<"[DEBUG] Env Obs Size " <<bEnvObs.Size() <<std::endl;
      }
      for(size_t i = 0; i < numRobots; ++i){
        bFailures<<failures[i];
        bRewards<<rewards[i];
      }
      for(size_t i = 0; i < stats.size(); ++i){
        bStats << stats[i];
      }
      std::cout<<"[SOCKET] Sending Multipart"<<std::endl;
      s.SendMsg(bState, true);
      std::cout<<"[DEBUG] State "<<bState<<" "<<bState.Size()<<std::endl;
      s.SendMsg(bEnvObs, true);
      std::cout<<"[DEBUG] Env Obs " <<bEnvObs<<" "<<bEnvObs.Size()<<std::endl;
      s.SendMsg(bFailures, true);
      std::cout<<"[DEBUG] Failures "<<bFailures<<" "<<bFailures.Size()<<std::endl;
      s.SendMsg(bRewards, true);
      std::cout<<"[DEBUG] Rewards "<<bRewards<<" "<< bRewards.Size()<<std::endl;
      s.SendMsg(bStats, false);
      std::cout<<"[DEBUG] Stats "<<bStats<<" "<<bStats.Size()<<std::endl;
      /*
      Receive Actions
      */

      CByteArray msg;
      s.RecvMsg(msg);
      std::cout<<"[Actions]";
      for(size_t i = 0; i < numRobots; ++i){
        std::vector<float> a;
        for(size_t j =0; j < 3; ++j){
          float action;
          getFloatIEEE754(msg, action);
          a.push_back(action);
          std::cout<<" "<< action;
        }
      }
      std::cout<<"\n";

      /*
      CByteArray a;
      a << 5;
      a << 6;
      a << 7;

      CByteArray b;
      b << 1 << 2<<3<<4;


      std::cout << "[INFO] Sending 5 6 7 (" << a.Size() << " bytes)" << std::endl;
      s.SendMsg(a, true);

      std::cout << "[INFO] Sending 1 2 3 4 (" << b.Size() << " bytes)" << std::endl;
      s.SendMsg(b, false);
      CByteArray c;
      s.RecvMsg(c);
      if(c[0] == 111 && c[1] == 107){
          std::cout<<"Ack"<<std::endl;
      }
      std::cout<<"[TEST] " << uint8_t(false)<<std::endl;
      */
   }
   catch(CARGoSException& ex) {
      std::cerr << ex.what() << std::endl;
      return 1;
   }
   return 0;
}
