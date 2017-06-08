#include <iostream>
#include <fstream>
#include <cstring>
#include <random>
#include <time.h>
#include <direct.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "slqLeNet5.h"


using namespace std;

namespace slqDL {
    slqLeNet5::~slqLeNet5()
    {
        if (trainImg)
        {
            delete trainImg;
            trainImg = nullptr;
        }
        if (trainLabel)
        {
            delete trainLabel;
            trainLabel = nullptr;
        }
        if (testImg)
        {
            delete testImg;
            testImg = nullptr;
        }
        if (testLabel)
        {
            delete testLabel;
            testLabel = nullptr;
        }

		if (trainData)
		{
			delete trainData;
			trainData = nullptr;
		}
		if (testData)
		{
			delete testData;
			testData = nullptr;
		}



    }

    void slqLeNet5::init()
    {
        ifstream trainimgstream;
        ifstream testimgstream;

        ifstream trainLabelStream;
        ifstream testLabelStream;

        initMap();

        trainimgstream.open("train-images.idx3-ubytepad", ifstream::binary | ifstream::in);
        trainimgstream.read(trainImg, TrainImgNum*imageRow*imageCol);
        trainimgstream.close();

        testimgstream.open("t10k-images.idx3-ubytepad", ifstream::binary | ifstream::in);
        testimgstream.read(testImg, TestImgNum*imageRow*imageCol);
        testimgstream.close();

        trainLabelStream.open("train-labels.idx1-ubytepad", ifstream::binary | ifstream::in);
        trainLabelStream.read(trainLabel, TrainImgNum);
        trainLabelStream.close();

        testLabelStream.open("t10k-labels.idx1-ubytepad", ifstream::binary | ifstream::in);
        testLabelStream.read(testLabel, TestImgNum);
        testLabelStream.close();

		inited = true;

    }

    void slqLeNet5::train()
    {
        if (false == inited)
            return;

        int eidx = 0;
        int lidx = 0;
        double precised = 0;
        double errored = 0;
        double cur_accur = 0;
		double currentErr;

		int Tab = 0;
		int TabCount = 0;
        for (eidx = 0; eidx < EpochLoop; eidx++)
        {

            cout << "epoch " << eidx << endl;
			QueryPerformanceFrequency(&CountFreq);
			QueryPerformanceCounter(&StartCount);
            for (lidx = 0; lidx < TrainImgNum; lidx++)
			{
				curLabelPtr = trainLabel + lidx;

				ProduceLabel();

				if (false == trainInit)
				{
					RegularMap(trainImg + lidx * inMapSize, trainData + lidx * inMapSize);
				}
                inmap = trainData + lidx * inMapSize;

				ForwardC1();
				ForwardS2();
				ForwardC3();
				ForwardS4();
				ForwardC5();
				ForwardOut();

				BackwardOut();
				BackwardC5();
				BackwardS4();
				BackwardC3();
				BackwardS2();
				BackwardC1();

				UpgradeNetwork();
                
            }

			QueryPerformanceCounter(&EndCount);
			cout << " time " << (double)(EndCount.QuadPart - StartCount.QuadPart) / CountFreq.QuadPart << endl;
			if (false == trainInit)
			{
				delete trainImg;
				trainImg = nullptr;
				trainInit = true;
			}



            cur_accur = test();

			cout << "current accuracy " << cur_accur << endl;

            if (cur_accur >= AccuracyRate)
            {
                cout << "current accuracy " << cur_accur << endl;
                SaveParameters();
                return;
            }
            
        }

        if (eidx == EpochLoop)
        {
            cout << "Non-precise current accuracy " << cur_accur << endl;
            SaveParameters();
        }
    }

	void slqLeNet5::predict()
	{
		HANDLE hfile;
		LPCTSTR lpFileName = L".\\*.png";
		WIN32_FIND_DATA pNextInfo;

		double imgArray[inMapSize];
		double *imgAPtr;
		char *imgBPtr;
		char imgName[20];
		char ch;

		double maxV = -10;
		int maxIdx = -1;

		
		ReadParameters();

		while (true)
		{

			hfile = FindFirstFile(lpFileName, &pNextInfo);
			if (INVALID_HANDLE_VALUE == hfile)
			{
				cout << "Error Handle to get First File" << endl;
				break;
			}

			while (FindNextFile(hfile, &pNextInfo))
			{
				cv::Mat image;
				for (hIdx = 0; hIdx < (wcslen(pNextInfo.cFileName) + 2); hIdx++)
				{
					imgName[hIdx] = pNextInfo.cFileName[hIdx];
				}
				imgName[hIdx] = '\0';
				cout << "image " << imgName << endl;
				image = cv::imread(imgName, CV_8UC1);
				if (image.empty())
				{
					cout << "Error image" << endl;
					return;
				}

				std::fill(imgArray, imgArray + inMapSize, -1);

				imgAPtr = imgArray;
				imgBPtr = (char*)image.data;

				double topV = -256;
				double botV = 256;

				for (hIdx = 0; hIdx < RawImgRow; hIdx++)
				{
					for (vIdx = 0; vIdx < RawImgCol; vIdx++)
					{
						int tmp = (int)(*(imgBPtr + hIdx*RawImgCol + vIdx));

						tmp = tmp < 0 ? (tmp + 256) : tmp;
						topV = topV > tmp ? topV : tmp;
						botV = botV < tmp ? botV : tmp;
					}
				}


				for (hIdx = 0; hIdx < RawImgRow; hIdx++)
				{
					for (vIdx = 0; vIdx < RawImgCol; vIdx++)
					{
						int tmp = (int)(*(imgBPtr + hIdx*RawImgCol + vIdx));

						tmp = tmp < 0 ? (tmp + 256) : tmp;
						*(imgAPtr + (hIdx + 2)*imageCol + (vIdx + 2)) = (tmp - botV) / (topV - botV) * 2.0 - 1.0;
					}
				}

				inmap = imgArray;
				ForwardC1();
				ForwardS2();
				ForwardC3();
				ForwardS4();
				ForwardC5();
				ForwardOut();

				maxV = -10;
				maxIdx = -1;
				for (hIdx = 0; hIdx < outMapSize; hIdx++)
				{
					maxV = maxV > outmap[hIdx] ? maxV : outmap[hIdx];
					maxIdx = maxV > outmap[hIdx] ? maxIdx : hIdx;
				}

				cout << "predict " << maxIdx << endl;

				ch = getchar();

				if ('!' == ch)
					return;

			}

		}

	}

	void slqLeNet5::checkimg()
	{
		ifstream testimgstream;
		testImg = new char[TestImgNum*inMapSize]();
		testimgstream.open("t10k-images.idx3-ubytepad", ifstream::binary | ifstream::in);
		testimgstream.read(testImg, TestImgNum*imageRow*imageCol);
		testimgstream.close();

		char imgName[20] = "testimg0.png";
		cv::Mat img(imageRow, imageCol, CV_8UC1);
		//for (hIdx = 0; hIdx < 30; hIdx++)
		//{
		//	sprintf_s(imgName, "testimg%d.png", hIdx + 1);
		//	memcpy((char*)img.data, testImg + hIdx*imageRow*imageCol, imageRow*imageCol);
		//	cv::imwrite(imgName, img);
		//}

		char imgArray[imageRow*imageCol];
		char tmpArray[imageRow*imageCol];
		memcpy((char*)img.data, testImg, imageRow*imageCol);
		memcpy(imgArray, testImg, imageRow*imageCol);
		cv::imwrite(imgName, img);

		cv::Mat rImg;
		rImg = cv::imread(imgName, CV_8UC1);
		memcpy(tmpArray, (char*)rImg.data, imageRow*imageCol);
		cv::imwrite("TestIMGL.png", rImg);

		for (hIdx = 0; hIdx < imageRow*4; hIdx++)
		{
			for (vIdx = 0; vIdx < imageCol/4; vIdx++)
			{
				char tmpc = imgArray[hIdx * (imageCol/4) + vIdx];
					int tmp = tmpc< 0 ? (tmpc + 256) : tmpc;
				cout << tmp << "\t";
			}
			cout << endl;

			for (vIdx = 0; vIdx < imageCol/4; vIdx++)
			{
				char tmpc = tmpArray[hIdx * (imageCol/4) + vIdx];
				int tmp = tmpc < 0 ? (tmpc + 256) : tmpc;
				cout << tmp << "\t";
			}
			cout << endl;
			cout << endl;
			cout << endl;
		}

		cv::imshow(imgName,rImg);
		cv::waitKey(0);
	}

    void slqLeNet5::initMap()
	{
		const double scale = 6.0;
		double min_;
		double max_;
		ifstream fileStream;

		fileStream.open("parameters_block", ifstream::in | ifstream::binary);

		if (!fileStream)
		{
			srand(time(0) + rand());

			min_ = -std::sqrt(scale / (25.0 + 150.0));
			max_ = std::sqrt(scale / (25.0 + 150.0));
			uniform_rand(c1conv, c1optNum, min_, max_);

			min_ = -std::sqrt(scale / (4.0 + 1.0));
			max_ = std::sqrt(scale / (4.0 + 1.0));
			uniform_rand(s2pool, s2MapNum, min_, max_);

			min_ = -std::sqrt(scale / (150.0 + 400.0));
			max_ = std::sqrt(scale / (150.0 + 400.0));
			uniform_rand(c3conv, c3optNum, min_, max_);

			min_ = -std::sqrt(scale / (4.0 + 1.0));
			max_ = std::sqrt(scale / (4.0 + 1.0));
			uniform_rand(s4pool, s4MapNum, min_, max_);

			min_ = -std::sqrt(scale / (400.0 + 3000.0));
			max_ = std::sqrt(scale / (400.0 + 3000.0));
			uniform_rand(c5conv, c5optNum, min_, max_);

			min_ = -std::sqrt(scale / (120.0 + 10.0));
			max_ = std::sqrt(scale / (120.0 + 10.0));
			uniform_rand(outfullconn, outoptNum, min_, max_);


			std::fill(c1bias, c1bias + c1MapNum, 0.0);
			std::fill(s2bias, s2bias + s2MapNum, 0.0);
			std::fill(c3bias, c3bias + c3MapNum, 0.0);
			std::fill(s4bias, s4bias + s4MapNum, 0.0);
			std::fill(c5bias, c5bias + c5MapNum, 0.0);
			std::fill(outbias, outbias + outMapSize, 0.0);
		}
		else
		{
			fileStream.close();
			ReadParameters();
		}


		std::fill(c1bias_Edt, c1bias_Edt + c1MapNum, 0.0);
		std::fill(c1conv_Edt, c1conv_Edt + c1optNum, 0.0);
		std::fill(s2bias_Edt, s2bias_Edt + s2MapNum, 0.0);
		std::fill(s2pool_Edt, s2pool_Edt + s2MapNum, 0.0);
		std::fill(c3bias_Edt, c3bias_Edt + c3MapNum, 0.0);
		std::fill(c3conv_Edt, c3conv_Edt + c3optNum, 0.0);
		std::fill(s4bias_Edt, s4bias_Edt + s4MapNum, 0.0);
		std::fill(s4pool_Edt, s4pool_Edt + s4MapNum, 0.0);
		std::fill(c5bias_Edt, c5bias_Edt + c5MapNum, 0.0);
		std::fill(c5conv_Edt, c5conv_Edt + c5optNum, 0.0);
		std::fill(outbias_Edt, outbias_Edt + outMapSize, 0.0);
		std::fill(outfull_Edt, outfull_Edt + outoptNum, 0.0);

        trainImg = new char[TrainImgNum*inMapSize]();
		trainLabel = new char[TrainImgNum]();
		testImg = new char[TestImgNum*inMapSize]();
		testLabel = new char[TestImgNum]();

		trainData = new double[TrainImgNum*inMapSize]();
		testData = new double[TestImgNum*inMapSize]();
    }

    void slqLeNet5::ForwardC1()
    {

        for (mIdx = 0; mIdx < c1MapNum; mIdx++)
        {
			layermulti = mIdx*f1MapSize;
            for (hIdx = 0; hIdx < c1MapHigh; hIdx++)
            {
				hmulti = hIdx*c1MapWidth;
                for (vIdx = 0; vIdx < c1MapWidth; vIdx++)
                {
                    double *curmap = c1map + layermulti + hmulti + vIdx;
                    *curmap = 0.f;

                    for (chIdx = 0; chIdx < convhsize; chIdx++)
                    {
                        for (cvIdx = 0; cvIdx < convwsize; cvIdx++)
                        {
                            *curmap += *(inmap + (hIdx + chIdx)*imageCol + vIdx + cvIdx) * (*(c1conv + mIdx*convsize + chIdx*convwsize + cvIdx));
                        }
                    }

                    *curmap += c1bias[mIdx];
                    *curmap = ACTIVATION(*curmap);
                }
            }
        }
    }

    void slqLeNet5::ForwardS2()
    {

        for (mIdx = 0; mIdx < s2MapNum; mIdx++)
        {
			layermulti = mIdx * f2MapSize;
            for (hIdx = 0; hIdx < s2MapHigh; hIdx++)
            {
				hmulti = hIdx*s2MapWidth;
                for (vIdx = 0; vIdx < s2MapWidth; vIdx++)
                {
                    double *curmap = s2map + layermulti + hmulti + vIdx;
                    double *c1cur = c1map + layermulti*4 + (hIdx * 2)*c1MapWidth + (vIdx * 2);
                    *curmap = s2bias[mIdx] + s2pool[mIdx] * (*c1cur + *(c1cur + 1) + *(c1cur + c1MapWidth) + *(c1cur + c1MapWidth + 1)) / 4.0;
                    *curmap = ACTIVATION(*curmap);
                }
            }
        }
    }

    void slqLeNet5::ForwardC3()
    {

        for (mIdx = 0; mIdx < c3MapNum; mIdx++)
        {
			layermulti = mIdx*f3MapSize;
            for (hIdx = 0; hIdx < c3MapHigh; hIdx++)
            {
				hmulti = hIdx * c3MapWidth;
                for (vIdx = 0; vIdx < c3MapWidth; vIdx++)
                {
                    double *curmap = c3map + layermulti + hmulti + vIdx;
                    *curmap = 0;

					for (int pIdx = 0; pIdx < s2MapNum; pIdx++)
					{
						if (!Twomap2ThreeTable[pIdx][mIdx])
							continue;

						double *curconv = c3conv + mIdx * s2MapNum * convsize + pIdx * convsize;
						for (chIdx = 0; chIdx < convhsize; chIdx++)
						{
							for (cvIdx = 0; cvIdx < convwsize; cvIdx++)
							{
								*curmap += *(s2map + pIdx * f2MapSize + (hIdx + chIdx)*s2MapWidth + (vIdx + cvIdx)) * (*(curconv + chIdx*convwsize + cvIdx));
							}
						}
					}


                    *curmap += c3bias[mIdx];
                    *curmap = ACTIVATION(*curmap);
                }
            }
        }
    }

    void slqLeNet5::ForwardS4()
    {

        for (mIdx = 0; mIdx < s4MapNum; mIdx++)
        {
            for (hIdx = 0; hIdx < s4MapHigh; hIdx++)
            {
                for (vIdx = 0; vIdx < s4MapWidth; vIdx++)
                {
                    double *curmap = s4map + mIdx*f4MapSize + hIdx*s4MapHigh + vIdx;
                    double *curc3 = c3map + mIdx*f3MapSize + (hIdx * 2)*c3MapWidth + (vIdx * 2);

                    *curmap = s4bias[mIdx] + s4pool[mIdx] * (*curc3 + *(curc3+1) + *(curc3+c3MapWidth) + *(curc3+c3MapWidth+1)) / 4.0;

                    *curmap = ACTIVATION(*curmap);
                }
            }
        }
    }

    void slqLeNet5::ForwardC5()
    {

        for (mIdx = 0; mIdx < c5MapNum; mIdx++)
        {
            for (hIdx = 0; hIdx < c5MapHigh; hIdx++)
            {
                for (vIdx = 0; vIdx < c5MapWidth; vIdx++)
                {
                    double *curmap = c5map + mIdx*c5MapHigh*c5MapWidth + hIdx*c5MapHigh + vIdx;
                    *curmap = 0;

					for (int pIdx = 0; pIdx < s4MapNum; pIdx++)
					{
						double *curconv = c5conv + mIdx * s4MapNum * convsize + pIdx * convsize;
						for (chIdx = 0; chIdx < convhsize; chIdx++)
						{
							for (cvIdx = 0; cvIdx < convwsize; cvIdx++)
							{
								*curmap += *(s4map + pIdx * f4MapSize + (hIdx + chIdx)*s4MapWidth + (vIdx + cvIdx)) * (*(curconv + chIdx*convhsize + cvIdx));
							}
						}
					}


                    *curmap += c5bias[mIdx];

                    *curmap = ACTIVATION(*curmap);
                }
            }
        }
    }

    void slqLeNet5::ForwardOut()
    {

        for (mIdx = 0; mIdx < outMapSize; mIdx++)
        {
            double *curmap = outmap + mIdx;
            *curmap = 0.f;

            for (hIdx = 0; hIdx < c5MapNum; hIdx++)
            {
                *curmap += c5map[hIdx] * outfullconn[hIdx*outMapSize+mIdx];
            }

            *curmap += outbias[mIdx];
            *curmap = ACTIVATION(*curmap);
        }
    }

    void slqLeNet5::BackwardOut()
    {

        for (mIdx = 0; mIdx < outMapSize; mIdx++)
        {
            outdelta[mIdx] = (outmap[mIdx] - label[mIdx]) * (ACTDEVICE(outmap[mIdx]));
            outbias_dt[mIdx] = outdelta[mIdx];
        }

        for (hIdx = 0; hIdx < c5MapNum; hIdx++)
        {
            for (vIdx = 0; vIdx < outMapSize; vIdx++)
            {
                outfull_dt[hIdx*outMapSize + vIdx] = c5map[hIdx] * outdelta[vIdx];
            }
        }
    }

    void slqLeNet5::BackwardC5()
    {

        for (mIdx = 0; mIdx < c5MapNum; mIdx++)
        {
            double curerr = 0;
            for (vIdx = 0; vIdx < outMapSize; vIdx++)
            {
                curerr += outdelta[vIdx] * outfullconn[mIdx*outMapSize + vIdx];
            }

            c5delta[mIdx] = ACTDEVICE(c5map[mIdx]) * curerr;

            c5bias_dt[mIdx] = c5delta[mIdx];

			for (int pIdx = 0; pIdx < s4MapNum; pIdx++)
			{
				double *curconv = c5conv_dt + mIdx * s4MapNum * convsize + pIdx*convsize;
				for (hIdx = 0; hIdx < s4MapHigh; hIdx++)
				{
					for (vIdx = 0; vIdx < s4MapWidth; vIdx++)
					{
						double *convdlt = curconv + hIdx*convwsize + vIdx;
						*convdlt = *(s4map + pIdx * f4MapSize + hIdx * convhsize + vIdx) * c5delta[mIdx];
					}
				}
			}
        }
    }

    void slqLeNet5::BackwardS4()
    {   

        for (mIdx = 0; mIdx < s4MapNum; mIdx++)
        {
			s4bias_dt[mIdx] = 0;
			s4pool_dt[mIdx] = 0;

			layermulti = mIdx*f4MapSize;
            for (hIdx = 0; hIdx < s4MapHigh; hIdx++)
            {
                for (vIdx = 0; vIdx < s4MapWidth; vIdx++)
                {
                    double *curdt = s4delta + layermulti + hIdx*s4MapWidth + vIdx;
                    double *curmap = s4map + layermulti + hIdx*s4MapWidth + vIdx;
                    double *c3cur = c3map + mIdx*f3MapSize + (hIdx * 2)*c3MapWidth + (vIdx * 2);

					*curdt = 0;

					for (int pIdx = 0; pIdx < c5MapNum; pIdx++)
					{
						double *curconv = c5conv + pIdx * s4MapNum * convsize + mIdx * convsize + hIdx * convhsize + vIdx;
						*curdt += *curconv * c5delta[pIdx];
					}
                    *curdt = ACTDEVICE(*curmap) * (*curdt);
					s4bias_dt[mIdx] += *curdt;

					s4pool_dt[mIdx] += (*c3cur + *(c3cur + 1) + *(c3cur + c3MapWidth) + *(c3cur + c3MapWidth + 1)) / 4.0 * (*curdt);
                }
            }
        }

    }

    void slqLeNet5::BackwardC3()
    {

        for (mIdx = 0; mIdx < c3MapNum; mIdx++)
        {
			c3bias_dt[mIdx] = 0;
			layermulti = mIdx*f3MapSize;
            for (hIdx = 0; hIdx < c3MapHigh; hIdx++)
            {
                for (vIdx = 0; vIdx < c3MapWidth; vIdx++)
                {
                    double *curdt = c3delta + layermulti + hIdx*c3MapWidth + vIdx;
                    double *curmap = c3map + layermulti + hIdx*c3MapWidth + vIdx;
                    double *s4dt = s4delta + mIdx*f4MapSize + (hIdx / 2)*s4MapWidth + (vIdx / 2);
                    *curdt = ACTDEVICE(*curmap)*s4pool[mIdx] * (*s4dt) / 4.0;

					c3bias_dt[mIdx] += *curdt;
                }
            }


			for (int pIdx = 0; pIdx < s2MapNum; pIdx++)
			{
				if (!Twomap2ThreeTable[pIdx][mIdx])
					continue;

				double *curcdt = c3conv_dt + mIdx * s2MapNum * convsize + pIdx * convsize;
				for (chIdx = 0; chIdx < convhsize; chIdx++)
				{
					for (cvIdx = 0; cvIdx < convwsize; cvIdx++)
					{
						double *curconv = curcdt + chIdx * convhsize + cvIdx;
						*curconv = 0;

						for (hIdx = 0; hIdx < c3MapHigh; hIdx++)
						{
							for (vIdx = 0; vIdx < c3MapWidth; vIdx++)
							{
								*curconv += *(s2map + pIdx * f2MapSize + (chIdx + hIdx)*s2MapWidth + (cvIdx + vIdx)) * c3delta[layermulti + hIdx*c3MapWidth + vIdx];
							}
						}
					}
				}
			}

        }
    }

    void slqLeNet5::BackwardS2()
    {

		memset((char*)s2delta, 0x0, s2MapSize*sizeof(double));
        for (mIdx = 0; mIdx < s2MapNum; mIdx++)
        {
            for (hIdx = 0; hIdx < c3MapHigh; hIdx++)
            {
                for (vIdx = 0; vIdx < c3MapWidth; vIdx++)
                {

					for (int pIdx = 0; pIdx < c3MapNum; pIdx++)
					{
						if (!Twomap2ThreeTable[mIdx][pIdx])
							continue;

						double *curconv = c3conv + pIdx * s2MapNum * convsize + mIdx * convsize;
						for (chIdx = 0; chIdx < convhsize; chIdx++)
						{
							for (cvIdx = 0; cvIdx < convwsize; cvIdx++)
							{
								double *curdt = s2delta + mIdx * f2MapSize + (hIdx + chIdx)*s2MapWidth + (vIdx + cvIdx);
								*curdt += *(curconv + chIdx * convwsize + cvIdx) * (*(c3delta + pIdx * f3MapSize + hIdx*c3MapWidth + vIdx));
							}
						}
					}
                }
            }
            s2bias_dt[mIdx] = 0;
            s2pool_dt[mIdx] = 0;
            for (hIdx = 0; hIdx < s2MapHigh; hIdx++)
            {
                for (vIdx = 0; vIdx < s2MapWidth; vIdx++)
                {
					double *curdt = s2delta + mIdx * f2MapSize + hIdx * s2MapWidth + vIdx;
					double *curmap = s2map + mIdx * f2MapSize + hIdx * s2MapWidth + vIdx;
                    double *curc1 = c1map+mIdx*f1MapSize+(hIdx*2)*c1MapWidth+(vIdx*2);

					*curdt *= ACTDEVICE(*curmap);
					s2bias_dt[mIdx] += *curdt;
                    s2pool_dt[mIdx] += *curdt * (*curc1 + *(curc1 + 1) + *(curc1 + c1MapWidth) + *(curc1 + c1MapWidth + 1)) / 4;
                }
            }
        }
    }

    void slqLeNet5::BackwardC1()
    {

        for (mIdx = 0; mIdx < c1MapNum; mIdx++)
        {
			c1bias_dt[mIdx] = 0;

			layermulti = mIdx*f1MapSize;
            for (hIdx = 0; hIdx < c1MapHigh; hIdx++)
            {
                for (vIdx = 0; vIdx < c1MapWidth; vIdx++)
                {
                    double *curdt = c1delta + layermulti + hIdx*c1MapWidth + vIdx;
                    double *curmap = c1map + layermulti + hIdx*c1MapWidth + vIdx;

                    *curdt = ACTDEVICE(*curmap) * s2pool[mIdx] * s2delta[layermulti/4 +(hIdx/2)*s2MapWidth+(vIdx/2)] / 4.0;

					c1bias_dt[mIdx] += *curdt;
                }
            }


            for (chIdx = 0; chIdx < convhsize; chIdx++)
            {
                for (cvIdx = 0; cvIdx < convwsize; cvIdx++)
                {
                    double *cdt = c1conv_dt + mIdx*convsize + chIdx*convwsize + cvIdx;
                    *cdt = 0;

                    for (hIdx = 0; hIdx < c1MapHigh; hIdx++)
                    {
                        for (vIdx = 0; vIdx < c1MapWidth; vIdx++)
                        {
                            *cdt += inmap[(hIdx + chIdx) * inMapWidth + (vIdx + cvIdx)] * c1delta[mIdx*c1MapHigh*c1MapWidth + hIdx*c1MapWidth + vIdx];
                        }
                    }
                }
            }
        }
    }

    void slqLeNet5::UpgradeNetwork()
    {
		UpdateParameters(c1bias_dt, c1bias_Edt, c1bias, c1MapNum);
		UpdateParameters(c1conv_dt, c1conv_Edt, c1conv, c1optNum);

		UpdateParameters(s2bias_dt, s2bias_Edt, s2bias, s2MapNum);
		UpdateParameters(s2pool_dt, s2pool_Edt, s2pool, s2MapNum);

		UpdateParameters(c3bias_dt, c3bias_Edt, c3bias, c3MapNum);
		UpdateParameters(c3conv_dt, c3conv_Edt, c3conv, c3optNum);

		UpdateParameters(s4bias_dt, s4bias_Edt, s4bias, s4MapNum);
		UpdateParameters(s4pool_dt, s4pool_Edt, s4pool, s4MapNum);

		UpdateParameters(c5bias_dt, c5bias_Edt, c5bias, c5MapNum);
		UpdateParameters(c5conv_dt, c5conv_Edt, c5conv, c5optNum);

		UpdateParameters(outbias_dt, outbias_Edt, outbias, outMapSize);
		UpdateParameters(outfull_dt, outfull_Edt, outfullconn, outoptNum);

    }

	void slqLeNet5::UpdateParameters(double *delta, double *Edelta, double *para, int len)
	{
		for (int lIdx = 0; lIdx < len; lIdx++)
		{
			Edelta[lIdx] += delta[lIdx] * delta[lIdx];
			para[lIdx] -= Alpha * delta[lIdx] / (std::sqrt(Edelta[lIdx]) + EspCNN);
		}
	}

    double slqLeNet5::test()
    {
        double preci = 0.f;
        double maxpred = 0.f;

		int tIdx;
        int oIdx;
        int dIdx;

        for (tIdx = 0; tIdx < TestImgNum; tIdx++)
        {
            tstLabelPtr = testLabel + tIdx;

			if (false == testInit)
				RegularMap(testImg + tIdx*inMapSize, testData + tIdx*inMapSize);

            inmap = testData + tIdx*inMapSize;

			ForwardC1();
			ForwardS2();
			ForwardC3();
			ForwardS4();
			ForwardC5();
			ForwardOut(); 
			
			maxpred = -10;
			for (oIdx = 0; oIdx < outMapSize; oIdx++)
			{
				maxpred = maxpred > outmap[oIdx] ? maxpred : outmap[oIdx];
				dIdx = maxpred > outmap[oIdx] ? dIdx : oIdx;
			}

			if (dIdx == (int)(*tstLabelPtr))
			{
				preci += 1.0; 
			}
        }

		if (false == testInit)
		{
			delete testImg;
			testImg = nullptr;
			testInit = true;
		}

        return (preci / TestImgNum);
    }

    void slqLeNet5::RandomBias(double *randVector, int vLen)
    {
        int tIdx = 0;
		for (; tIdx < vLen; tIdx++)
		{
			randVector[tIdx] = 0;
		}
    }


    void slqLeNet5::SaveParameters()
    {
        ofstream fileStream;

        fileStream.open("parameters_block", ofstream::out | ofstream::binary);

		fileStream.write((char*)c1conv, sizeof(double)*c1optNum);
		fileStream.write((char*)c1bias, sizeof(double)*c1MapNum);
		fileStream.write((char*)s2pool, sizeof(double)*s2MapNum);
		fileStream.write((char*)s2bias, sizeof(double)*s2MapNum);
		fileStream.write((char*)c3conv, sizeof(double)*c3optNum);
		fileStream.write((char*)c3bias, sizeof(double)*c3MapNum);
		fileStream.write((char*)s4pool, sizeof(double)*s4MapNum);
		fileStream.write((char*)s4bias, sizeof(double)*s4MapNum);
		fileStream.write((char*)c5conv, sizeof(double)*c5optNum);
		fileStream.write((char*)c5bias, sizeof(double)*c5MapNum);
		fileStream.write((char*)outfullconn, sizeof(double)*outoptNum);
		fileStream.write((char*)outbias, sizeof(double)*outMapSize);

        fileStream.close();
    }

	void slqLeNet5::ReadParameters()
	{
		ifstream fileStream;

		fileStream.open("parameters_block", ifstream::in | ifstream::binary);

		fileStream.read((char*)c1conv, sizeof(double)*c1optNum);
		fileStream.read((char*)c1bias, sizeof(double)*c1MapNum);
		fileStream.read((char*)s2pool, sizeof(double)*s2MapNum);
		fileStream.read((char*)s2bias, sizeof(double)*s2MapNum);
		fileStream.read((char*)c3conv, sizeof(double)*c3optNum);
		fileStream.read((char*)c3bias, sizeof(double)*c3MapNum);
		fileStream.read((char*)s4pool, sizeof(double)*s4MapNum);
		fileStream.read((char*)s4bias, sizeof(double)*s4MapNum);
		fileStream.read((char*)c5conv, sizeof(double)*c5optNum);
		fileStream.read((char*)c5bias, sizeof(double)*c5MapNum);
		fileStream.read((char*)outfullconn, sizeof(double)*outoptNum);
		fileStream.read((char*)outbias, sizeof(double)*outMapSize);

		fileStream.close();
	}

	void slqLeNet5::ProduceLabel()
	{
		int tIdx;

		for (tIdx = 0; tIdx < outMapSize; tIdx++)
		{
			if (tIdx == *curLabelPtr)
				label[tIdx] = 0.8f;
			else
				label[tIdx] = -0.8f;
		}
	}

	void slqLeNet5::RegularMap(char *cmap, double *rmap)
	{
		for (hIdx = 0; hIdx < imageRow; hIdx++)
		{
			for (vIdx = 0; vIdx < imageCol; vIdx++)
			{
				int tmp = (int)(*(cmap + hIdx*imageCol + vIdx));

				tmp = tmp < 0 ? (tmp + 256) : tmp;
				*(rmap + hIdx*imageCol + vIdx) = tmp / 255.0 * 2 - 1;
			}
		}
	}

	void slqLeNet5::uniform_rand(double* src, int len, double min, double max)
	{
		for (int i = 0; i < len; i++) {
			src[i] = uniform_rand(min, max);
		}

	}

	double slqLeNet5::uniform_rand(double min, double max)
	{
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<double> dst(min, max);
		return dst(gen);
	}

}
