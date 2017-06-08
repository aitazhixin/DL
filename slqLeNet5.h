#ifndef __SLQ_LE_NET_5_H__
#define __SLQ_LE_NET_5_H__

#include <cmath>
#include <cstdlib>
#include <Windows.h>
/**
* LeNet5 Neural Network Define
**/

namespace slqDL {
/** Train Parameters **/
#define EpochLoop           (100)              // Train Loops
#define AccuracyRate        (0.99)             // Train Accuracy
#define Alpha               (0.01)             // Train Step Length
#define LoopError           (0.001)            //
#define EspCNN              (1e-8)             //

	/** Raw Data Parameters **/
#define TrainImgNum          (60000)            // Train Image Number
#define TestImgNum           (10000)            // Test Image Number
#define RawImgRow            (28)               // Raw Image Row
#define RawImgCol            (28)               // Raw Image Col
#define imageRow             (32)               // Image Row
#define imageCol             (32)               // Image Col

    /** Convolution or Pooling Parameters **/
#define convhsize            (5)                // Convolution factor high size
#define convwsize            (5)                // Convolution factor width size
#define convsize             (25)               // Convolution factor size
#define poolhsize            (2)                // Pooling factor high size
#define poolwsize            (2)                // Pooling factor width size

    /** Feature Map Parameters **/
    /** Input map Parameters **/
#define inMapHigh            (imageRow)         // input map high: equal to imageRow 32
#define inMapWidth           (imageCol)         // input map Width: equal to imageCol 32
#define inMapSize            (1024)             // inMapHigh * inMapWidth: 32*32

    /** C1 Layer Parameters **/
#define c1MapHigh            (28)               //
#define c1MapWidth           (28)               //
#define f1MapSize            (784)              //
#define c1MapNum             (6)                //
#define c1optNum             (150)              // 6 * 5 * 5
#define c1MapSize            (4704)             //

    /** S2 Layer Parameters **/
#define s2MapHigh            (14)               //
#define s2MapWidth           (14)               //
#define f2MapSize            (196)              //
#define s2MapNum             (6)                //
#define s2MapSize            (1176)             //

    /** C3 Layer Parameters **/
#define c3MapHigh            (10)               //
#define c3MapWidth           (10)               //
#define f3MapSize            (100)              //
#define c3MapNum             (16)               //
#define c3optNum             (2400)             // 16 * 6 * 5 * 5
#define c3MapSize            (1600)             //

    /** S4 Layer Parameters **/
#define s4MapHigh            (5)                //
#define s4MapWidth           (5)                //
#define f4MapSize            (25)               //
#define s4MapNum             (16)               //
#define s4MapSize            (400)              //

    /** C5 Layer Parameters **/
#define c5MapHigh            (1)                //
#define c5MapWidth           (1)                //
#define c5MapNum             (120)              //
#define c5optNum             (48000)            // 120 * 16 * 5 * 5
#define c5MapSize            (120)              //

    /** Output Map Parameters **/
#define outMapSize           (10)               //
#define outoptNum            (1200)             //

#define TabNum               (10)               //

#define ACTIVATION(x)       ((std::exp((x)) - std::exp(-1*(x))) / (std::exp((x)) + std::exp(-1*(x))))
#define ACTDEVICE(x)        ((1 - ((x))*((x))))

#define O                   (true)
#define X                   (false)

	static const bool Twomap2ThreeTable[s2MapNum][c3MapNum] =
	{
		O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O,
		O, O, X, X, X, O, O, O, X, X, O, O, O, O, X, O,
		O, O, O, X, X, X, O, O, O, X, X, O, X, O, O, O,
		X, O, O, O, X, X, O, O, O, O, X, X, O, X, O, O,
		X, X, O, O, O, X, X, O, O, O, O, X, O, O, X, O,
		X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O
	};

	//static const bool Fourmap2FiveTabel[s4MapNum][c5MapNum];

class slqLeNet5
{
public:
    slqLeNet5() = default;
    slqLeNet5(const slqLeNet5 &lenet) = default;
    slqLeNet5 & operator = (const slqLeNet5 &lenet) = default;
    ~slqLeNet5();

    void init();
    void train();
	void predict();
	void checkimg();

private:
    void initMap();
    void ForwardC1();
    void ForwardS2();
    void ForwardC3();
    void ForwardS4();
    void ForwardC5();
    void ForwardOut();

    void BackwardOut();
    void BackwardC5();
    void BackwardS4();
    void BackwardC3();
    void BackwardS2();
    void BackwardC1();

    void UpgradeNetwork();
	void UpdateParameters(double *delta, double *Edelta, double *para, int len);

    double test();

    void SaveParameters();
	void ReadParameters();
    void RandomBias(double *randVector, int vLen);
	void ProduceLabel();
	void RegularMap(char *cmap, double *mapdata);

	void uniform_rand(double* src, int len, double min, double max);
	double uniform_rand(double min, double max);


private:


	double c1map[c1MapSize];
	double c1bias[c1MapNum];
	double c1conv[c1optNum];
	double c1bias_dt[c1MapNum];
	double c1bias_Edt[c1MapNum];
	double c1conv_dt[c1optNum];
	double c1conv_Edt[c1optNum];
	double c1delta[c1MapSize];

	double s2map[s2MapSize];
	double s2bias[s2MapNum];
	double s2pool[s2MapNum];
	double s2bias_dt[s2MapNum];
	double s2bias_Edt[s2MapNum];
	double s2pool_dt[s2MapNum];
	double s2pool_Edt[s2MapNum];
	double s2delta[s2MapSize];

	double c3map[c3MapSize];
	double c3bias[c3MapNum];
	double c3conv[c3optNum];
	double c3bias_dt[c3MapNum];
	double c3bias_Edt[c3MapNum];
	double c3conv_dt[c3optNum];
	double c3conv_Edt[c3optNum];
	double c3delta[c3MapSize];

	double s4map[s4MapSize];
	double s4bias[s4MapNum];
	double s4pool[s4MapNum];
	double s4bias_dt[s4MapNum];
	double s4bias_Edt[s4MapNum];
	double s4pool_dt[s4MapNum];
	double s4pool_Edt[s4MapNum];
	double s4delta[s4MapSize];

	double c5map[c5MapSize];
	double c5bias[c5MapNum];
	double c5conv[c5optNum];
	double c5bias_dt[c5MapNum];
	double c5bias_Edt[c5MapNum];
	double c5conv_dt[c5optNum];
	double c5conv_Edt[c5optNum];
	double c5delta[c5MapSize];

	double outmap[outMapSize];
	double outbias[outMapSize];
	double outfullconn[outoptNum];
	double outbias_dt[outMapSize];
	double outbias_Edt[outMapSize];
	double outfull_dt[outoptNum];
	double outfull_Edt[outoptNum];
	double outdelta[outMapSize];

	double label[outMapSize];

    char *trainImg;
    char *trainLabel;
    char *testImg;
    char *testLabel;

	double *trainData;
	double *testData;

	double *inmap;
    char *tstLabelPtr;
    char *curLabelPtr;
    bool inited = false;
	bool trainInit = false;
	bool testInit = false;

	int mIdx;
	int hIdx;
	int vIdx;
	int chIdx;
	int cvIdx;

	int layermulti;
	int hmulti;

	LARGE_INTEGER StartCount;
	LARGE_INTEGER EndCount;
	LARGE_INTEGER CountFreq;
};
} // end namespace slqDL

#endif



