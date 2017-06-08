
#include <iostream>

#include "slqLeNet5.h"

using namespace std;
using namespace slqDL;

int main()
{
    slqLeNet5 cnnNet;

    cout << "Init CNN Net :" << endl;
    cnnNet.init();

    cout << "Training ..." << endl;
    cnnNet.train();

	//cout << "predict ..." << endl;
	//cnnNet.predict();

	//cout << "checking ..." << endl;
	//cnnNet.checkimg();


	return 0;
}
