
#include <iostream>
#include "slqAlexNet.h"

using namespace std;
using namespace slqDL::slqAlexNet;

int main()
{
    slqAlexNet alexNet;

    cout << "init ..." << endl;
    alexNet.init();
    return 0;
}
