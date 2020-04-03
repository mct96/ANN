#include "../include/activation_function.hpp"

namespace ann {

ActivationFunction::~ActivationFunction()
{

}

LinearFunction::LinearFunction(double a)
    :
    ActivationFunction{},
    _a{a}
{

}

LinearFunction::~LinearFunction()
{

}

double LinearFunction::f(double x) const
{
    return _a * x;
}

double LinearFunction::Df(double x) const
{
    return _a;
}

StepFunction::StepFunction(double yUpper, double yLower, double x0)
    :
    ActivationFunction{},
    _yUpper{yUpper},
    _yLower{yLower},
    _x0{x0}
{

}

StepFunction::~StepFunction() {}

double StepFunction::f(double x) const
{
    return x >= _x0 ? _yUpper : _yLower;
}

double StepFunction::Df(double x) const
{
    return 0;
}

ArctanFunction::ArctanFunction()
    :
    ActivationFunction{}
{

}

ArctanFunction::~ArctanFunction()
{

}

double ArctanFunction::f(double x) const
{
    return atan(x);
}

double ArctanFunction::Df(double x) const
{
    return 1/(1 + x*x);
}

HyperbolicTangentFunction::HyperbolicTangentFunction()
    :
    ActivationFunction{}
{

}

HyperbolicTangentFunction::~HyperbolicTangentFunction()
{

}

double HyperbolicTangentFunction::f(double x) const
{
    return tanh(x);
}

double HyperbolicTangentFunction::Df(double x) const
{
    double z = cosh(x);
    return 1/(z*z); // sech^2(x)
}

LogisticFunction::LogisticFunction(double a)
    :
    ActivationFunction{},
    _a{a}
{

}

LogisticFunction::~LogisticFunction()
{

}

double LogisticFunction::f(double x) const
{
    return 1/(1+std::exp(-_a*x));
}

double LogisticFunction::Df(double x) const
{
    return f(x) * (1 - f(x));
}

}