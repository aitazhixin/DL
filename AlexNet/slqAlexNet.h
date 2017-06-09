#ifndef __SLQ_ALEX_NET_H__
#define __SLQ_ALEX_NET_H__

#include "slqNetMacro.h"

namespace slqDL {
    namespace slqAlexNet{
        class slqAlexNet {
        public:
            slqAlexNet() = default;
            slqAlexNet(const slqAlexNet &slqNet) = default;
            slqAlexNet & operator = (const slqAlexNet & slqNet) = default;
            ~slqAlexNet();
            
            void init();
            void train();
            void predict();
            
        private:
            void deletevar(float **var);
            void initParm();
            void newParam();
            void ForwardC1();
            void ForwardS1();
            void ForwardC2();
            void ForwardS2();
            void ForwardC3();
            void ForwardC4();
            void ForwardC5();
            void ForwardS5();
            void ForwardF1();
            void ForwardF2();
            void ForwardF3();

            void BackwardF3();
            void BackwardF2();
            void BackwardF1();
            void BackwardS5();
            void BackwardC5();
            void BackwardC4();
            void BackwardC3();
            void BackwardS2();
            void BackwardC2();
            void BackwardS1();
            void BackwardC1();

            void UpgradeNetwork();
            void UpdateParameters(float *delta, float *Edelta, float *para, int len);

            float test();
            
            void SaveParameters();
            void ReadParameters();
            void RandomBias(float *randVector, int vLen);
            void ProduceLabel();
            void RegularMap(char *cmap, float *mapdata);

            void uniform_rand(float* src, int len, float min, float max);
            float uniform_rand(float min, float max);
            
        private:
            float *mlabel;
            float *inMap;
            float *c1Map;
            float *s1Map;
            float *c2Map;
            float *s2Map;
            float *c3Map;
            float *c4Map;
            float *c5Map;
            float *s5Map;
            float *f1Map;
            float *f2Map;
            float *f3Map;
            
            float *c1Conv;
            float *s1Pool;
            float *c2Conv;
            float *s2Pool;
            float *c3Conv;
            float *c4Conv;
            float *c5Conv;
            float *s5Pool;
            float *f1Conn;
            float *f2Conn;
            float *f3Conn;
            
            float *c1Bias;
            float *s1Bias;
            float *c2Bias;
            float *s2Bias;
            float *c3Bias;
            float *c4Bias;
            float *c5Bias;
            float *s5Bias;
            float *f1Bias;
            float *f2Bias;
            float *f3Bias;
            
            float *c1MapDt;
            float *s1MapDt;
            float *c2MapDt;
            float *s2MapDt;
            float *c3MapDt;
            float *c4MapDt;
            float *c5MapDt;
            float *s5MapDt;
            float *f1MapDt;
            float *f2MapDt;
            float *f3MapDt;
            
            float *c1ConvDt;
            float *s1PoolDt;
            float *c2ConvDt;
            float *s2PoolDt;
            float *c3ConvDt;
            float *c4ConvDt;
            float *c5ConvDt;
            float *s5PoolDt;
            float *f1ConnDt;
            float *f2ConnDt;
            float *f3ConnDt;
            
            float *c1BiasDt;
            float *s1BiasDt;
            float *c2BiasDt;
            float *s2BiasDt;
            float *c3BiasDt;
            float *c4BiasDt;
            float *c5BiasDt;
            float *s5BiasDt;
            float *f1BiasDt;
            float *f2BiasDt;
            float *f3BiasDt;
            
            float *c1ConvEDt;
            float *s1PoolEDt;
            float *c2ConvEDt;
            float *s2PoolEDt;
            float *c3ConvEDt;
            float *c4ConvEDt;
            float *c5ConvEDt;
            float *s5PoolEDt;
            float *f1ConnEDt;
            float *f2ConnEDt;
            float *f3ConnEDt;
            
            float *c1BiasEDt;
            float *s1BiasEDt;
            float *c2BiasEDt;
            float *s2BiasEDt;
            float *c3BiasEDt;
            float *c4BiasEDt;
            float *c5BiasEDt;
            float *s5BiasEDt;
            float *f1BiasEDt;
            float *f2BiasEDt;
            float *f3BiasEDt;
            
        };
    } // end namespace slqAlexNet
} // end namespace slqDL

#endif // #ifndef __SLQ_ALEX_NET_H__
