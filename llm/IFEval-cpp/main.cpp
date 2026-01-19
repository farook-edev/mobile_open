#include <iostream>
#include "ifeval.h"


int main(){
    std::string t(
        (std::istreambuf_iterator<char>(std::cin)),
        std::istreambuf_iterator<char>()
        );

    std::cout << "string size: " << t.size() << std::endl;

    mlperf::mobile::IFEval dataset(t);

    dataset.ComputeAccuracyString();

    return 0;
}
