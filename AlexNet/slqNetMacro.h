// AlexNet define
#ifndef __SLQ_NET_MACRO_H__
#define __SLQ_NET_MACRO_H__

#include <cmath>
namespace slqDL {
  namespace slqAlexNet{
    
#define EpochLoop           (100)
#define AccuracyRate        (0.99)
#define Alpha               (0.01)
#define LoopError           (0.001)
#define EspCNN              (1e-8)
    
#define poolSpace                    (3)
#define poolStride                   (2)
    
#define inMapHigh                    (227)
#define inMapWidth                   (227)
#define inMapNum                     (3)
#define inMapSize                    (51529)                         // 227*227
#define inUnitNum                    (154587)                        // 227*227*3
    
#define c1MapHigh                    (55)
#define c1MapWidth                   (55)
#define c1MapNum                     (96)
#define c1MapSize                    (3025)                          // 55*55
#define c1UnitNum                    (290400)                        // 55*55*96
    
#define c1ConvHigh                   (11)
#define c1ConvWidth                  (11)
#define c1ConvDeep                   (3)
#define c1ConvStride                 (4)
#define c1ConvSize                   (363)                            // 11*11*3
#define c1ConvNum                    (96)
#define c1ConvUNum                   (34848)                          // 11*11*3*96
    
#define s1MapHigh                    (27)
#define s1MapWidth                   (27)
#define s1MapNum                     (96)
#define s1MapSize                    (729)                           // 27*27
#define s1UnitNum                    (69984)                         // 27*27*96
    
#define c2MapHigh                    (27)
#define c2MapWidth                   (27)
#define c2MapNum                     (256)
#define c2MapSize                    (729)                           // 27*27
#define c2UnitNum                    (186624)                        // 27*27*256
    
#define c2ConvHigh                   (5)
#define c2ConvWidth                  (5)
#define c2ConvDeep                   (48)                            // 96/2
#define c2ConvStride                 (1)
#define c2ConvSize                   (1200)                          // 5*5*48
#define c2ConvNum                    (256)
#define c2ConvUNum                   (307200)                        // 5*5*48*256
    
#define s2MapHigh                    (13)
#define s2MapWidth                   (13)
#define s2MapNum                     (256)
#define s2MapSize                    (169)                           // 13*13
#define s2UnitNum                    (43264)                         // 13*13*256
    
#define c3MapHigh                    (13)
#define c3MapWidth                   (13)
#define c3MapNum                     (384)
#define c3MapSize                    (169)                           // 13*13
#define c3UnitNum                    (64896)                         // 13*13*384
    
#define c3ConvHigh                   (3)
#define c3ConvWidth                  (3)
#define c3ConvDeep                   (256)
#define c3ConvStride                 (1)
#define c3ConvSize                   (2304)                          // 3*3*256
#define c3ConvNum                    (384)
#define c3ConvUNum                   (884736)                        // 3*3*256*384
    

#define c4MapHigh                    (13)
#define c4MapWidth                   (13)
#define c4MapNum                     (384)
#define c4MapSize                    (169)                           // 13*13
#define c4UnitNum                    (64896)                         // 13*13*384
    
#define c4ConvHigh                   (3)
#define c4ConvWidth                  (3)
#define c4ConvDeep                   (192)                           // 384/2
#define c4ConvStride                 (1)
#define c4ConvSize                   (1728)                          // 3*3*192
#define c4ConvNum                    (384)
#define c4ConvUNum                   (663552)                        // 3*3*192*384
    
#define c5MapHigh                    (13)
#define c5MapWidth                   (13)
#define c5MapNum                     (256)
#define c5MapSize                    (169)                           // 13*13
#define c5UnitNum                    (43264)                         // 13*13*256
    
#define c5ConvHigh                   (3)
#define c5ConvWidth                  (3)
#define c5ConvDeep                   (192)                           // 384/2
#define c5ConvStride                 (1)
#define c5ConvSize                   (1728)                          // 3*3*192
#define c5ConvNum                    (256)
#define c5ConvUNum                   (442368)                        // 3*3*192*256
    
#define s5MapHigh                    (6)
#define s5MapWidth                   (6)
#define s5MapNum                     (256)
#define s5MapSize                    (36)                            // 6*6
#define s5UnitNum                    (9216)                          // 6*6*256
    
#define f1UnitNum                    (4096)
#define f1ConnNum                    (37748736)                      // 6*6*256*4096
    
#define f2UnitNum                    (4096)
#define f2ConnNum                    (16777216)                      // 4096*4096
    
#define f3UnitNum                    (1000)
#define f3ConnNum                    (4096000)                       // 4096*1000
    
#define ACTIVATION(x)                ((std::exp((x)) - std::exp(-1*(x))) / (std::exp((x)) + std::exp(-1*(x))))
#define ACTDEVICE(x)                 ((1 - ((x))*((x))))
    
  } // end namespace slqAlexNet
} // end namespace slqDL


#endif // #ifndef __SLQ_NET_MACRO_H__

