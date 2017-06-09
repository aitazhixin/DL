

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "slqAlexNet.h"

using namespace std;

namespace slqDL {
    namespace slqAlexNet {

slqAlexNet::~slqAlexNet()
{
    deletevar(&mlabel);
    deletevar(&inMap);
    deletevar(&c1Map);
    deletevar(&s1Map);
    deletevar(&c2Map);
    deletevar(&s2Map);
    deletevar(&c3Map);
    deletevar(&c4Map);
    deletevar(&c5Map);
    deletevar(&s5Map);
    deletevar(&f1Map);
    deletevar(&f2Map);
    deletevar(&f3Map);
            
    deletevar(&c1Conv);
    deletevar(&s1Pool);
    deletevar(&c2Conv);
    deletevar(&s2Pool);
    deletevar(&c3Conv);
    deletevar(&c4Conv);
    deletevar(&c5Conv);
    deletevar(&s5Pool);
    deletevar(&f1Conn);
    deletevar(&f2Conn);
    deletevar(&f3Conn);
            
    deletevar(&c1Bias);
    deletevar(&s1Bias);
    deletevar(&c2Bias);
    deletevar(&s2Bias);
    deletevar(&c3Bias);
    deletevar(&c4Bias);
    deletevar(&c5Bias);
    deletevar(&s5Bias);
    deletevar(&f1Bias);
    deletevar(&f2Bias);
    deletevar(&f3Bias);
            
    deletevar(&c1MapDt);
    deletevar(&s1MapDt);
    deletevar(&c2MapDt);
    deletevar(&s2MapDt);
    deletevar(&c3MapDt);
    deletevar(&c4MapDt);
    deletevar(&c5MapDt);
    deletevar(&s5MapDt);
    deletevar(&f1MapDt);
    deletevar(&f2MapDt);
    deletevar(&f3MapDt);
            
    deletevar(&c1ConvDt);
    deletevar(&s1PoolDt);
    deletevar(&c2ConvDt);
    deletevar(&s2PoolDt);
    deletevar(&c3ConvDt);
    deletevar(&c4ConvDt);
    deletevar(&c5ConvDt);
    deletevar(&s5PoolDt);
    deletevar(&f1ConnDt);
    deletevar(&f2ConnDt);
    deletevar(&f3ConnDt);
            
    deletevar(&c1BiasDt);
    deletevar(&s1BiasDt);
    deletevar(&c2BiasDt);
    deletevar(&s2BiasDt);
    deletevar(&c3BiasDt);
    deletevar(&c4BiasDt);
    deletevar(&c5BiasDt);
    deletevar(&s5BiasDt);
    deletevar(&f1BiasDt);
    deletevar(&f2BiasDt);
    deletevar(&f3BiasDt);
            
    deletevar(&c1ConvEDt);
    deletevar(&s1PoolEDt);
    deletevar(&c2ConvEDt);
    deletevar(&s2PoolEDt);
    deletevar(&c3ConvEDt);
    deletevar(&c4ConvEDt);
    deletevar(&c5ConvEDt);
    deletevar(&s5PoolEDt);
    deletevar(&f1ConnEDt);
    deletevar(&f2ConnEDt);
    deletevar(&f3ConnEDt);
            
    deletevar(&c1BiasEDt);
    deletevar(&s1BiasEDt);
    deletevar(&c2BiasEDt);
    deletevar(&s2BiasEDt);
    deletevar(&c3BiasEDt);
    deletevar(&c4BiasEDt);
    deletevar(&c5BiasEDt);
    deletevar(&s5BiasEDt);
    deletevar(&f1BiasEDt);
    deletevar(&f2BiasEDt);
    deletevar(&f3BiasEDt);
}


void slqAlexNet::init()
{
    initParm();
}


void slqAlexNet::train()
{

    ForwardC1();
    ForwardS1();
    ForwardC2();
    ForwardS2();
    ForwardC3();
    ForwardC4();
    ForwardC5();
    ForwardS5();
    ForwardF1();
    ForwardF2();
    ForwardF3();

    BackwardF3();
    BackwardF2();
    BackwardF1();
    BackwardS5();
    BackwardC5();
    BackwardC4();
    BackwardC3();
    BackwardS2();
    BackwardC2();
    BackwardS1();
    BackwardC1();

    UpgradeNetwork();

    test();
}


void slqAlexNet::deletevar(float **var)
{
    if (nullptr != *var)
    {
        delete *var;
        *var = nullptr;
    }
}


void slqAlexNet::initParm()
{
    newParm();

    // all weights follow gauss distribution
    uniform_rand(c1Conv, c1ConvUNum, 0.f, 0.f);
    uniform_rand(s1Pool, s1MapNum, 0.f, 0.f);
    uniform_rand(c2Conv, c2ConvUNum, 0.f, 0.f);
    uniform_rand(s2Pool, s2MapNum, 0.f, 0.f);
    uniform_rand(c3Conv, c3ConvUNum, 0.f, 0.f);
    uniform_rand(c4Conv, c4ConvUNum, 0.f, 0.f);
    uniform_rand(c5Conv, c5ConvUNum, 0.f, 0.f);
    uniform_rand(s5Pool, s5MapNum, 0.f, 0.f);
    uniform_rand(f1Conn, f1ConnNum, 0.f, 0.f);
    uniform_rand(f2Conn, f2ConnNum, 0.f, 0.f);
    uniform_rand(f3Conn, f3ConnNum, 0.f, 0.f);

    // 2th, 4th, 5th conv bias set to 1
    std::fill(c2Bias, c2Bias + c2MapNum, 1.0f);
    std::fill(c4Bias, c4Bias + c4MapNum, 1.0f);
    std::fill(c5Bias, c5Bias + c5MapNum, 1.0f);


}


void slqAlexNet::newParm()
{
    mlabel = new float[f3UnitNum]();
    inMap = new float[inUnitNum]();
    c1Map = new float[c1UnitNum]();
    s1Map = new float[s1UnitNum]();
    c2Map = new float[c2UnitNum]();
    s2Map = new float[s2UnitNum]();
    c3Map = new float[c3UnitNum]();
    c4Map = new float[c4UnitNum]();
    c5Map = new float[c5UnitNum]();
    s5Map = new float[s5UnitNum]();
    f1Map = new float[f1UnitNum]();
    f2Map = new float[f2UnitNum]();
    f3Map = new float[f3UnitNum]();

    c1Conv = new float[c1ConvUNum]();
    s1Pool = new float[s1MapNum]();
    c2Conv = new float[c2ConvUNum]();
    s2Pool = new float[s2MapNum]();
    c3Conv = new float[c3ConvUNum]();
    c4Conv = new float[c4ConvUNum]();
    c5Conv = new float[c5ConvUNum]();
    s5Pool = new float[s5MapNum]();
    f1Conn = new float[f1ConnNum]();
    f2Conn = new float[f2ConnNum]();
    f3Conn = new float[f3ConnNum]();

    c1Bias = new float[c1MapNum]();
    s1Bias = new float[s1MapNum]();
    c2Bias = new float[c2MapNum]();
    s2Bias = new float[s2MapNum]();
    c3Bias = new float[c3MapNum]();
    c4Bias = new float[c4MapNum]();
    c5Bias = new float[c5MapNum]();
    s5Bias = new float[s5MapNum]();
    f1Bias = new float[f1UnitNum]();
    f2Bias = new float[f2UnitNum]();
    f3Bias = new float[f3UnitNum]();

    c1MapDt = new float[c1UnitNum]();
    s1MapDt = new float[s1UnitNum]();
    c2MapDt = new float[c2UnitNum]();
    s2MapDt = new float[s2UnitNum]();
    c3MapDt = new float[c3UnitNum]();
    c4MapDt = new float[c4UnitNum]();
    c5MapDt = new float[c5UnitNum]();
    s5MapDt = new float[s5UnitNum]();
    f1MapDt = new float[f1UnitNum]();
    f2MapDt = new float[f2UnitNum]();
    f3MapDt = new float[f3UnitNum]();

    c1ConvDt = new float[c1ConvUNum]();
    s1PoolDt = new float[s1MapNum]();
    c2ConvDt = new float[c2ConvUNum]();
    s2PoolDt = new float[s2MapNum]();
    c3ConvDt = new float[c3ConvUNum]();
    c4ConvDt = new float[c4ConvUNum]();
    c5ConvDt = new float[c5ConvUNum]();
    s5PoolDt = new float[s5MapNum]();
    f1ConnDt = new float[f1ConnNum]();
    f2ConnDt = new float[f2ConnNum]();
    f3ConnDt = new float[f3ConnNum]();

    c1BiasDt = new float[c1MapNum]();
    s1BiasDt = new float[s1MapNum]();
    c2BiasDt = new float[c2MapNum]();
    s2BiasDt = new float[s2MapNum]();
    c3BiasDt = new float[c3MapNum]();
    c4BiasDt = new float[c4MapNum]();
    c5BiasDt = new float[c5MapNum]();
    s5BiasDt = new float[s5MapNum]();
    f1BiasDt = new float[f1UnitNum]();
    f2BiasDt = new float[f2UnitNum]();
    f3BiasDt = new float[f3UnitNum]();

    c1ConvEDt = new float[c1ConvUNum]();
    s1PoolEDt = new float[s1MapNum]();
    c2ConvEDt = new float[c2ConvUNum]();
    s2PoolEDt = new float[s2MapNum]();
    c3ConvEDt = new float[c3ConvUNum]();
    c4ConvEDt = new float[c4ConvUNum]();
    c5ConvEDt = new float[c5ConvUNum]();
    s5PoolEDt = new float[s5MapNum]();
    f1ConnEDt = new float[f1ConnNum]();
    f2ConnEDt = new float[f2ConnNum]();
    f3ConnEDt = new float[f3ConnNum]();

    c1BiasEDt = new float[c1MapNum]();
    s1BiasEDt = new float[s1MapNum]();
    c2BiasEDt = new float[c2MapNum]();
    s2BiasEDt = new float[s2MapNum]();
    c3BiasEDt = new float[c3MapNum]();
    c4BiasEDt = new float[c4MapNum]();
    c5BiasEDt = new float[c5MapNum]();
    s5BiasEDt = new float[s5MapNum]();
    f1BiasEDt = new float[f1UnitNum]();
    f2BiasEDt = new float[f2UnitNum]();
    f3BiasEDt = new float[f3UnitNum]();
}


void slqAlexNet::ForwardC1()
{
    int odeepIdx = 0;
    int ideepIdx = 0;
    int ohIdx = 0;
    int owIdx = 0;
    int ihIdx = 0;
    int iwIdx = 0;
    int cvhIdx = 0;
    int cvwIdx = 0;

    int odh;
    int iod;
    int vdh;
    int ivd;

    std::fill(c1Map, c1Map + c1UnitNum, 0.f);
    for (odeepIdx = 0; odeepIdx < c1MapNum; odeepIdx++)
    {
        for (ohIdx = 0; ohIdx < c1MapHigh; ohIdx++)
        {
            odh = odeepIdx * c1MapSize + ohIdx * c1MapWidth;
            for (owIdx = 0; owIdx < c1MapWidth; owIdx++)
            {
                float *curmap = c1Map + odh + owIdx;                                     // (h, w, d) first convolution output
                for (ideepIdx = 0; ideepIdx < c1ConvDeep; ideepIdx++)
                {
                    iod = odeepIdx * c1ConvSize + ideepIdx * c1ConvHigh * c1ConvWidth;
                    for (cvhIdx = 0; cvhIdx < c1ConvHigh; cvhIdx++)
                    {
                        vdh = iod + cvhIdx * c1ConvWidth;
                        ivd = ideepIdx * inMapSize + (ohIdx * c1ConvStride + cvhIdx) * c1ConvWidth + owIdx * c1ConvStride;
                        for (cvwIdx = 0; cvwIdx < c1ConvWidth; cvwIdx++)
                        {
                            float *curin = inMap + ivd + cvwIdx;

                            *curmap += *curin * c1Conv[vdh + cvwIdx];
                        }
                    }
                }

                *curmap += c1Bias[odeepIdx];

                *curmap = ACTIVATION(*curmap);
            }
        }
    }
}


void slqAlexNet::ForwardS1()
{
    int odeepIdx;
    int ohIdx;
    int owIdx;
    int shIdx;
    int swIdx;

    int odh;
    int sdh;

    std::fill(s1Map, s1Map + s1UnitNum, 0.f);
    for (odeepIdx = 0; odeepIdx < s1MapNum; odeepIdx++)
    {
        for (ohIdx = 2; ohIdx < s1MapHigh-2; ohIdx++)
        {
            odh = odeepIdx * s1MapSize + ohIdx * s1MapWidth;
            sdh = odeepIdx * c1MapSize + (ohIdx-2) * poolStride * c1MapWidth;
            for (owIdx = 2; owIdx < s1MapWidth-2; owIdx++)
            {
                float *curmap = s1Map + odh + owIdx;
                float *curin = c1Map + sdh + (owIdx-2) * poolStride;
                for (shIdx = 0; shIdx < poolSpace; shIdx++)
                {
                    for (swIdx = 0; swIdx < poolSpace; swIdx++)
                    {
                        *curmap += *curin + shIdx * c1MapWidth + swIdx;
                    }
                }

                *curmap = *curmap / (poolSpace * poolSpace) * s1Pool[odeepIdx] + s1Bias[odeepIdx];
                *curmap = ACTIVATION(*curmap);
            }
        }
    }
}


void slqAlexNet::ForwardC2()
{
    int odeepIdx;
    int ohIdx;
    int owIdx;
    int ideepIdx;
    int cvhIdx;
    int cvwIdx;

    int odh;
    int ioh;
    int vod;

    std::fill(c2Map, c2Map + c2UnitNum, 0.f);
    for (odeepIdx = 0; odeepIdx < c2MapNum / 2; odeepIdx++)
    {
        for (ohIdx = 0; ohIdx < c2MapHigh; ohIdx++)
        {
            odh = odeepIdx * c2MapSize + ohIdx * c2MapHigh;
            for (owIdx = 0; owIdx < c2MapWidth; owIdx++)
            {
                float *curmap = c2Map + odh + owIdx;
                for (ideepIdx = 0; ideepIdx < c2ConvDeep; ideepIdx++)
                {
                    ioh = odeepIdx * c2ConvSize + ideepIdx * c2ConvHigh * c2ConvWidth;
                    float *curin = s1Map + ideepIdx * s1MapSize + ohIdx * s1MapWidth + owIdx;

                    for (cvhIdx = 0; cvhIdx < c2ConvHigh; cvhIdx++)
                    {
                        vod = ioh + cvhIdx * c2ConvWidth;
                        for (cvwIdx = 0; cvwIdx < c2ConvWidth; cvwIdx++)
                        {
                            *curmap += *(curin + cvhIdx * s1MapWidth + cvwIdx) * c2Conv[vod + cvwIdx];
                        }
                    }
                }

                *curmap += c2Bias[odeepIdx];
                *curmap = ACTIVATION(*curmap);
            }
        }
    }

    for (odeepIdx = c2MapNum / 2; odeepIdx < c2MapNum; odeepIdx++)
    {
        for (ohIdx = 0; ohIdx < c2MapHigh; ohIdx++)
        {
            odh = odeepIdx * c2MapSize + ohIdx * c2MapHigh;
            for (owIdx = 0; owIdx < c2MapWidth; owIdx++)
            {
                float *curmap = c2Map + odh + owIdx;
                for (ideepIdx = c2ConvDeep; ideepIdx < s1MapNum; ideepIdx++)
                {
                    ioh = odeepIdx * c2ConvSize + (ideepIdx - c2ConvDeep) * c2ConvHigh * c2ConvWidth;
                    float *curin = s1Map + ideepIdx * s1MapSize + ohIdx * s1MapWidth + owIdx;

                    for (cvhIdx = 0; cvhIdx < c2ConvHigh; cvhIdx++)
                    {
                        vod = ioh + cvhIdx * c2ConvWidth;
                        for (cvwIdx = 0; cvwIdx < c2ConvWidth; cvwIdx++)
                        {
                            *curmap += *(curin + cvhIdx * s1MapWidth + cvwIdx) * c2Conv[vod + cvwIdx];
                        }
                    }
                }

                *curmap += c2Bias[odeepIdx];
                *curmap = ACTIVATION(*curmap);
            }
        }
    }
}


void slqAlexNet::ForwardS2()
{
    int odeepIdx;
    int ohIdx;
    int owIdx;
    int shIdx;
    int swIdx;

    int odh;
    int sdh;

    std::fill(s2Map, s2Map + s2UnitNum, 0.f);
    for (odeepIdx = 0; odeepIdx < s2MapNum; odeepIdx++)
    {
        for (ohIdx = 1; ohIdx < s2MapHigh - 1; ohIdx++)
        {
            odh = odeepIdx * s2MapSize + ohIdx * s2MapWidth;
            sdh = odeepIdx * c2MapSize + (ohIdx - 1) * poolStride * c2MapWidth;

            for (owIdx = 1; owIdx < s2MapWidth - 1; owIdx++)
            {
                float *curmap = s2Map + odh + owIdx;
                float *curin = c2Map + sdh + (owIdx - 1) * poolStride;
                for (shIdx = 0; shIdx < poolSpace; shIdx++)
                {
                    for (swIdx = 0; swIdx < poolSpace; swIdx++)
                    {
                        *curmap += *curin + shIdx * c2MapWidth + swIdx;
                    }
                }

                *curmap = *curmap / (poolSpace * poolSpace) * s2Pool[odeepIdx] + s2Bias[odeepIdx];
                *curmap = ACTIVATION(*curmap);
            }
        }
    }
}


void slqAlexNet::ForwardC3()
{
    int odeepIdx;
    int ohIdx;
    int owIdx;
    int ideepIdx;
    int cvhIdx;
    int cvwIdx;

    int ohd;
    int iod;
    int vod;

    std::fill(c3Map, c3Map + c3UnitNum, 0.f);
    for (odeepIdx = 0; odeepIdx < c3MapNum; odeepIdx++)
    {
        for (ohIdx = 1; ohIdx < c3MapHigh - 1; ohIdx++)
        {
            ohd = odeepIdx * c3MapSize + ohIdx * c3MapWidth;
            for (owIdx = 1; owIdx < c3MapWidth - 1; owIdx++)
            {
                float *curmap = c3Map + ohd + owIdx;
                for (ideepIdx = 0; ideepIdx < c3ConvDeep; ideepIdx++)
                {
                    float *curin = s2Map + ideepIdx * s2MapSize + (ohIdx - 1) * s2MapWidth + owIdx;
                    iod = odeepIdx * c3ConvSize + ideepIdx * c3ConvHigh * c3ConvWidth;

                    for (cvhIdx = 0; cvhIdx < c3ConvHigh; cvhIdx++)
                    {
                        vod = iod + cvhIdx * c3ConvWidth;
                        for (cvwIdx = 0; cvwIdx < c3ConvWidth; cvwIdx++)
                        {
                            *curmap += *(curin + cvhIdx * s2MapWidth + cvwIdx) * c3Conv[vod + cvwIdx];
                        }
                    }
                }

                *curmap += c3Bias[odeepIdx];

                *curmap = ACTIVATION(*curmap);
            }
        }
    }

}


void slqAlexNet::ForwardC4()
{
    int odeepIdx;
    int ohIdx;
    int owIdx;
    int ideepIdx;
    int cvhIdx;
    int cvwIdx;

    int ohd;
    int iod;
    int vod;

    std::fill(c4Map, c4Map + c4UnitNum, 0.f);
    for (odeepIdx = 0; odeepIdx < c4MapNum/2; odeepIdx++)
    {
        for (ohIdx = 1; ohIdx < c4MapHigh - 1; ohIdx++)
        {
            ohd = odeepIdx * c4MapSize + ohIdx * c4MapWidth;
            for (owIdx = 1; owIdx < c4MapWidth - 1; owIdx++)
            {
                float *curmap = c4Map + ohd + owIdx;
                for (ideepIdx = 0; ideepIdx < c4ConvDeep; ideepIdx++)
                {
                    float *curin = c3Map + ideepIdx * c3MapSize + (ohIdx - 1) * c3MapWidth + owIdx;
                    iod = odeepIdx * c4ConvSize + ideepIdx * c4ConvHigh * c4ConvWidth;

                    for (cvhIdx = 0; cvhIdx < c4ConvHigh; cvhIdx++)
                    {
                        vod = iod + cvhIdx * c4ConvWidth;
                        for (cvwIdx = 0; cvwIdx < c4ConvWidth; cvwIdx++)
                        {
                            *curmap += *(curin + cvhIdx * c3MapWidth + cvwIdx) * c4Conv[vod + cvwIdx];
                        }
                    }
                }

                *curmap += c4Bias[odeepIdx];

                *curmap = ACTIVATION(*curmap);
            }
        }
    }

    for (odeepIdx = c4MapNum/2; odeepIdx < c4MapNum; odeepIdx++)
    {
        for (ohIdx = 1; ohIdx < c4MapHigh - 1; ohIdx++)
        {
            ohd = odeepIdx * c4MapSize + ohIdx * c4MapWidth;
            for (owIdx = 1; owIdx < c4MapWidth - 1; owIdx++)
            {
                float *curmap = c4Map + ohd + owIdx;
                for (ideepIdx = c4ConvDeep; ideepIdx < c3MapNum; ideepIdx++)
                {
                    float *curin = c3Map + ideepIdx * c3MapSize + (ohIdx - 1) * c3MapWidth + owIdx;
                    iod = odeepIdx * c4ConvSize + (ideepIdx - c4ConvDeep) * c4ConvHigh * c4ConvWidth;

                    for (cvhIdx = 0; cvhIdx < c4ConvHigh; cvhIdx++)
                    {
                        vod = iod + cvhIdx * c4ConvWidth;
                        for (cvwIdx = 0; cvwIdx < c4ConvWidth; cvwIdx++)
                        {
                            *curmap += *(curin + cvhIdx * c3MapWidth + cvwIdx) * c4Conv[vod + cvwIdx];
                        }
                    }
                }

                *curmap += c4Bias[odeepIdx];

                *curmap = ACTIVATION(*curmap);
            }
        }
    }
}


void slqAlexNet::ForwardC5()
{
    int odeepIdx;
    int ohIdx;
    int owIdx;
    int ideepIdx;
    int cvhIdx;
    int cvwIdx;

    int ohd;
    int iod;
    int vod;

    std::fill(c5Map, c5Map + c5UnitNum, 0.f);
    for (odeepIdx = 0; odeepIdx < c5MapNum/2; odeepIdx++)
    {
        for (ohIdx = 0; ohIdx < c5MapHigh; ohIdx++)
        {
            ohd = odeepIdx * c5MapSize + ohIdx * c5MapWidth;
            for (owIdx = 0; owIdx < c5MapWidth; owIdx++)
            {
                float *curmap = c5Map + ohd + owIdx;
                for (ideepIdx = 0; ideepIdx < c5ConvDeep; ideepIdx++)
                {
                    float *curin = c4Map + ideepIdx * c4MapSize + (ohIdx - 1) * c4MapWidth + owIdx;
                    iod = odeepIdx * c5ConvSize + ideepIdx * c5ConvHigh * c5ConvWidth;

                    for (cvhIdx = 0; cvhIdx < c5ConvHigh; cvhIdx++)
                    {
                        vod = iod + cvhIdx * c5ConvWidth;
                        for (cvwIdx = 0; cvwIdx < c5ConvWidth; cvwIdx++)
                        {
                            *curmap += *(curin + cvhIdx * c4MapWidth + cvwIdx) * c5Conv[vod + cvwIdx];
                        }
                    }
                }

                *curmap += c5Bias[odeepIdx];

                *curmap = ACTIVATION(*curmap);
            }
        }
    }

    for (odeepIdx = c5MapNum / 2; odeepIdx < c5MapNum; odeepIdx++)
    {
        for (ohIdx = 0; ohIdx < c5MapHigh; ohIdx++)
        {
            ohd = odeepIdx * c5MapSize + ohIdx * c5MapWidth;
            for (owIdx = 0; owIdx < c5MapWidth; owIdx++)
            {
                float *curmap = c5Map + ohd + owIdx;
                for (ideepIdx = c5ConvDeep; ideepIdx < c4MapNum; ideepIdx++)
                {
                    float *curin = c4Map + ideepIdx * c4MapSize + (ohIdx - 1) * c4MapWidth + owIdx;
                    iod = odeepIdx * c5ConvSize + (ideepIdx - c5ConvDeep) * c5ConvHigh * c5ConvWidth;

                    for (cvhIdx = 0; cvhIdx < c5ConvHigh; cvhIdx++)
                    {
                        vod = iod + cvhIdx * c5ConvWidth;
                        for (cvwIdx = 0; cvwIdx < c5ConvWidth; cvwIdx++)
                        {
                            *curmap += *(curin + cvhIdx * c4MapWidth + cvwIdx) * c5Conv[vod + cvwIdx];
                        }
                    }
                }

                *curmap += c5Bias[odeepIdx];

                *curmap = ACTIVATION(*curmap);
            }
        }
    }
}


void slqAlexNet::ForwardS5()
{
    int odeepIdx;
    int ohIdx;
    int owIdx;
    int phIdx;
    int pwIdx;

    int odh;
    int sdh;

    std::fill(s5Map, s5Map + s5UnitNum, 0.f);
    for (odeepIdx = 0; odeepIdx < s5MapNum; odeepIdx++)
    {
        for (ohIdx = 0; ohIdx < s5MapHigh; ohIdx++)
        {
            odh = odeepIdx * s5MapSize + ohIdx * s5MapWidth;
            sdh = odeepIdx * c5MapSize + ohIdx * poolStride * c5MapWidth;

            for (owIdx = 1; owIdx < s5MapWidth - 1; owIdx++)
            {
                float *curmap = s5Map + odh + owIdx;
                float *curin = c5Map + sdh + owIdx * poolStride;
                for (phIdx = 0; phIdx < poolSpace; phIdx++)
                {
                    for (pwIdx = 0; pwIdx < poolSpace; pwIdx++)
                    {
                        *curmap += *curin + phIdx * c5MapWidth + pwIdx;
                    }
                }

                *curmap = *curmap / (poolSpace * poolSpace) * s5Pool[odeepIdx] + s5Bias[odeepIdx];
                *curmap = ACTIVATION(*curmap);
            }
        }
    }
}


void slqAlexNet::ForwardF1()
{
    int oIdx;
    int iIdx;

    int ood;

    std::fill(f1Map, f1Map + f1UnitNum, 0.f);
    for (oIdx = 0; oIdx < f1UnitNum; oIdx++)
    {
        float *curmap = f1Map + oIdx;
        ood = oIdx * s5UnitNum;
        for (iIdx = 0; iIdx < s5UnitNum; iIdx++)
        {
            *curmap += s5Map[iIdx] * f1Conn[ood + iIdx];
        }

        *curmap += f1Bias[oIdx];
        *curmap = ACTIVATION(*curmap);
    }
}


void slqAlexNet::ForwardF2()
{
    int oIdx;
    int iIdx;

    int iod;

    std::fill(f2Map, f2Map + f2UnitNum, 0.f);
    for (oIdx = 0; oIdx < f2UnitNum; oIdx++)
    {
        float *curmap = f2Map + oIdx;
        iod = oIdx * f1UnitNum;
        for (iIdx = 0; iIdx < f1UnitNum; iIdx++)
        {
            *curmap += f1Map[iIdx] * f2Conn[iod + iIdx];
        }

        *curmap += f2Bias[oIdx];
        *curmap = ACTIVATION(*curmap);
    }
}


void slqAlexNet::ForwardF3()
{
    int oIdx;
    int iIdx;

    int iod;

    std::fill(f3Map, f3Map + f3UnitNum, 0.f);
    for (oIdx = 0; oIdx < f3UnitNum; oIdx++)
    {
        float *curmap = f3Map + oIdx;
        iod = oIdx * f2UnitNum;
        for (iIdx = 0; iIdx < f2UnitNum; iIdx++)
        {
            *curmap += f2Map[iIdx] * f3Conn[iod + iIdx];
        }

        *curmap += f3Bias[oIdx];
        *curmap = ACTIVATION(*curmap);
    }
}

void slqAlexNet::BackwardF3()
{
    int oIdx;
    int iIdx;

    for (oIdx = 0; oIdx < f3UnitNum; oIdx++)
    {
        f3MapDt[oIdx] = (f3Map[oIdx] - mlabel[oIdx]) * ACTDEVICE(f3Map[oIdx]);
        f3BiasDt[oIdx] = f3MapDt[oIdx];

        for (iIdx = 0; iIdx < f2UnitNum; iIdx++)
        {
            f3ConnDt[iIdx*f3UnitNum + oIdx] = f3MapDt[oIdx] * f2Map[iIdx];
        }
    }
}


void slqAlexNet::BackwardF2()
{
    int oIdx;
    int iIdx;

    for (oIdx = 0; oIdx < f2UnitNum; oIdx++)
    {
        float curdt = 0.f;
        for (iIdx = 0; iIdx < f3UnitNum; iIdx++)
        {
            curdt += f3MapDt[iIdx] * f3Conn[oIdx * f3UnitNum + iIdx];
        }

        f2MapDt[oIdx] = ACTDEVICE(f2Map[oIdx]) * curdt;
        f2BiasDt[oIdx] = f2MapDt[oIdx];

        for (iIdx = 0; iIdx < f1UnitNum; iIdx++)
        {
            f2ConnDt[iIdx * f2UnitNum + oIdx] = f1Map[iIdx] * f2MapDt[oIdx];
        }
    }
}


void slqAlexNet::BackwardF1()
{
    int oIdx;
    int iIdx;

    for (oIdx = 0; oIdx < f1UnitNum; oIdx++)
    {
        float curdt = 0.f;
        for (iIdx = 0; iIdx < f2UnitNum; iIdx++)
        {
            curdt += f2MapDt[iIdx] * f2Conn[oIdx * f2UnitNum + iIdx];
        }

        f1MapDt[oIdx] = ACTDEVICE(f1Map[oIdx]) * curdt;
        f1BiasDt[oIdx] = f1MapDt[oIdx];

        for (iIdx = 0; iIdx < s5UnitNum; iIdx++)
        {
            f1ConnDt[iIdx * f1UnitNum + oIdx] = f1MapDt[oIdx] * s5Map[iIdx];
        }
    }
}


void slqAlexNet::BackwardS5()
{
    int oIdx;
    int iIdx;
    int iod;

    int odeepIdx;
    int ohIdx;
    int owIdx;
    int phIdx;
    int pwIdx;

    std::fill(s5BiasDt, s5BiasDt + s5MapNum, 0.f);
    std::fill(s5PoolDt, s5PoolDt + s5MapNum, 0.f);
    for (oIdx = 0; oIdx < s5UnitNum; oIdx++)
    {
        float curdt = 0.f;
        iod = oIdx * f1UnitNum;
        for (iIdx = 0; iIdx < f1UnitNum; iIdx++)
        {
            curdt += f1MapDt[iIdx] * f1Conn[iod + iIdx];
        }

        s5MapDt[oIdx] = ACTDEVICE(s5Map[oIdx]) * curdt;
    }

    for (odeepIdx = 0; odeepIdx < s5MapNum; odeepIdx++)
    {
        for (ohIdx = 0; ohIdx < s5MapHigh; ohIdx++)
        {
            iod = odeepIdx*s5MapSize + ohIdx * s5MapWidth;
            for (owIdx = 0; owIdx < s5MapWidth; owIdx++)
            {
                s5BiasDt[odeepIdx] += s5MapDt[iod + owIdx];

                float *curin = c5Map + odeepIdx * c5MapSize + ohIdx * poolStride * c5MapWidth + owIdx * poolStride;
                float curdt = 0.f;
                for (phIdx = 0; phIdx < poolSpace; phIdx++)
                {
                    for (pwIdx = 0; pwIdx < poolSpace; pwIdx++)
                    {
                        curdt += *(curin + phIdx * c5MapWidth + pwIdx);
                    }
                }

                s5PoolDt[odeepIdx] += curdt / (poolSpace * poolSpace) * s5MapDt[iod + owIdx];
            }
        }
    }
}


void slqAlexNet::BackwardC5()
{
    int odeepIdx;
    int ideepIdx;
    int ohIdx;
    int owIdx;
    int phIdx;
    int pwIdx;

    int iod;
    int vod;
    int svh;

    int top;

    std::fill(c5MapDt, c5MapDt + c5UnitNum, 0.f);
    std::fill(c5BiasDt, c5BiasDt + c5MapNum, 0.f);
    std::fill(c5ConvDt, c5ConvDt + c5ConvUNum, 0.f);

    for (odeepIdx = 0; odeepIdx < c5MapNum; odeepIdx)
    {
        for (ohIdx = 0; ohIdx < s5MapHigh; ohIdx++)
        {
            for (owIdx = 0; owIdx < s5MapWidth; owIdx++)
            {
                for (phIdx = 0; phIdx < poolSpace; phIdx++)
                {
                    for (pwIdx = 0; pwIdx < poolSpace; pwIdx++)
                    {
                        iod = odeepIdx * c5MapSize + (ohIdx + poolStride + phIdx) * c5MapWidth + (owIdx + poolStride + pwIdx);
                        vod = odeepIdx * s5MapSize + ohIdx * s5MapWidth + owIdx;
                        c5MapDt[iod] += ACTDEVICE(c5Map[iod]) * s5Pool[odeepIdx] * s5MapDt[vod] / (poolSpace * poolSpace);
                        c5BiasDt[odeepIdx] += c5MapDt[iod];
                    }
                }
                
            }
        }

        top = odeepIdx > c5MapNum / 2 ? 0 : 1;

        for (ideepIdx = 0; ideepIdx < c5ConvDeep; ideepIdx++)
        {
            for (phIdx = 0; phIdx < c5ConvHigh; phIdx++)
            {
                for (pwIdx = 0; pwIdx < c5ConvWidth; pwIdx++)
                {
                    iod = odeepIdx * c5ConvSize + ideepIdx * c5ConvHigh * c5ConvWidth + phIdx * c5ConvWidth + pwIdx;
                    float *curconv = c5ConvDt + iod;

                    for (ohIdx = 0; ohIdx < c5MapHigh; ohIdx++)
                    {
                        for (owIdx = 0; owIdx < c5MapWidth; owIdx++)
                        {
                            svh = (ideepIdx + top * c5ConvDeep) * c4MapSize + (phIdx + ohIdx * c5ConvStride)*c4MapWidth + (pwIdx + owIdx * c5ConvStride);
                            vod = odeepIdx * c5MapSize + ohIdx * c5MapWidth + owIdx;
                            *curconv += c4Map[svh] * c5MapDt[vod];
                        }
                    }
                }
            }
        }
    }
}


void slqAlexNet::BackwardC4()
{
    int odeepIdx;
    int ideepIdx;
    int ohIdx;
    int owIdx;
    int phIdx;
    int pwIdx;

    int iod;
    int vod;
    int svh;

    int top;

    std::fill(c4MapDt, c4MapDt + c4UnitNum, 0.f);
    std::fill(c4BiasDt, c4BiasDt + c4MapNum, 0.f);
    std::fill(c4ConvDt, c4ConvDt + c4ConvUNum, 0.f);
    for (odeepIdx = 0; odeepIdx < c4MapNum / 2; odeepIdx++)
    {
        for (ideepIdx = 0; ideepIdx < c5MapNum / 2; ideepIdx++)
        {
            for (phIdx = 0; phIdx < c5ConvHigh; phIdx++)
            {
                for (pwIdx = 0; pwIdx < c5ConvWidth; pwIdx++)
                {
                    for (ohIdx = 0; ohIdx < c5MapHigh; ohIdx++)
                    {
                        for (owIdx = 0; owIdx < c5MapWidth; owIdx++)
                        {
                            iod = odeepIdx * c4MapSize + (phIdx + ohIdx * c5ConvStride) * c4MapWidth + (pwIdx + owIdx * c5ConvStride);
                            vod = ideepIdx * c5MapSize + ohIdx * c5MapWidth + owIdx;
                            svh = ideepIdx * c5ConvSize + odeepIdx * c5ConvHigh * c5ConvWidth + phIdx * c5ConvWidth + pwIdx;
                            c4MapDt[iod] += c5Conv[svh] * c5MapDt[vod];
                            c4BiasDt[odeepIdx] += c4MapDt[iod];
                        }
                    }
                }
            }
        }
    }

    for (odeepIdx = c4MapNum / 2; odeepIdx < c4MapNum; odeepIdx++)
    {
        for (ideepIdx = c5MapNum / 2; ideepIdx < c5MapNum; ideepIdx++)
        {
            for (phIdx = 0; phIdx < c5ConvHigh; phIdx++)
            {
                for (pwIdx = 0; pwIdx < c5ConvWidth; pwIdx++)
                {
                    for (ohIdx = 0; ohIdx < c5MapHigh; ohIdx++)
                    {
                        for (owIdx = 0; owIdx < c5MapWidth; owIdx++)
                        {
                            iod = odeepIdx * c4MapSize + (phIdx + ohIdx * c5ConvStride) * c4MapWidth + (pwIdx + owIdx * c5ConvStride);
                            vod = ideepIdx * c5MapSize + ohIdx * c5MapWidth + owIdx;
                            svh = ideepIdx * c5ConvSize + (odeepIdx - c5ConvDeep) * c5ConvHigh * c5ConvWidth + phIdx * c5ConvWidth + pwIdx;
                            c4MapDt[iod] += c5Conv[svh] * c5MapDt[vod];
                            c4BiasDt[odeepIdx] += c4MapDt[iod];
                        }
                    }
                }
            }
        }
    }

    for (odeepIdx = 0; odeepIdx < c4MapNum; odeepIdx++)
    {
        for (ideepIdx = 0; ideepIdx < c4ConvDeep; ideepIdx++)
        {
            for (phIdx = 0; phIdx < c4ConvHigh; phIdx++)
            {
                for (pwIdx = 0; pwIdx < c4ConvWidth; pwIdx++)
                {

                }
            }
        }
    }


}


void slqAlexNet::BackwardC3()
{}


void slqAlexNet::BackwardS2()
{}


void slqAlexNet::BackwardC2()
{}


void slqAlexNet::BackwardS1()
{}


void slqAlexNet::BackwardC1()
{}



void slqAlexNet::UpgradeNetwork()
{}


void slqAlexNet::UpdateParameters(float *delta, float *Edelta, float *para, int len)
{
    for (int lIdx = 0; lIdx < len; lIdx++)
    {
        Edelta[lIdx] += delta[lIdx] * delta[lIdx];
        para[lIdx] -= Alpha * delta[lIdx] / (std::sqrt(Edelta[lIdx]) + EspCNN);
    }
}


float slqAlexNet::test()
{
    return 0;
}


void slqAlexNet::SaveParameters()
{}


void slqAlexNet::ReadParameters()
{}


void slqAlexNet::RandomBias(float *randVector, int vLen)
{}


void slqAlexNet::ProduceLabel()
{}


void slqAlexNet::RegularMap(char *cmap, float *mapdata)
{}


void slqAlexNet::uniform_rand(float* src, int len, float min, float max)
{
    int rIdx = 0;

    for (; rIdx < len; rIdx++)
    {
        src[rIdx] = uniform_rand(min, max);
    }
}


float slqAlexNet::uniform_rand(float min, float max)
{
    static float U, V;
    static int phase = 0;
    float gaussz = 0;

    if (0 == phase)
    {
        U = rand() / (RAND_MAX + 1.0f);
        V = rand() / (RAND_MAX + 1.0f);

        gaussz = 0.1f * std::sqrt(-2.0f * log(U)) * sin(2.0f * CV_PI * V);
    }
    else
    {
        gaussz = 0.1f * std::sqrt(-2.0f * log(U)) * cos(2.0f * CV_PI * V);
    }

    phase = 1 - phase;
    return gaussz;
}



} // end namespace slqAlexNet

} // end namespace slqDL
