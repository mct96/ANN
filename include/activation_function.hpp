#pragma once

#include <cmath>

namespace ann {

class ActivationFunction {
public:
    virtual ~ActivationFunction();

    virtual double f(double x) const = 0;
    virtual double Df(double x) const = 0;
};


class LinearFunction: public ActivationFunction {
public:
    LinearFunction(double a);
    virtual ~LinearFunction();

    virtual double f(double x) const;
    virtual double Df(double x) const;
private:
    double _a;
};

class StepFunction: public ActivationFunction {
public:
    StepFunction(double yUpper, double yLower, double x0);
    virtual ~StepFunction();

    virtual double f(double x) const;
    virtual double Df(double x) const;

private:
    double _x0;
    double _yUpper;
    double _yLower;
};

class ArctanFunction: public ActivationFunction {
public:
    ArctanFunction();
    virtual ~ArctanFunction();

    virtual double f(double x) const;
    virtual double Df(double x) const;
};

class HyperbolicTangentFunction: public ActivationFunction {
public:
    HyperbolicTangentFunction();
    virtual ~HyperbolicTangentFunction();

    virtual double f(double x) const;
    virtual double Df(double x) const;
};

class LogisticFunction: public ActivationFunction  {
public:
    LogisticFunction(double a);
    virtual ~LogisticFunction();

    double f(double x) const override;
    double Df(double x) const override;

private:
    double _a;
};

}