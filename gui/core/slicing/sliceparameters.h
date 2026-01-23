#ifndef SLICEPARAMETERS_H
#define SLICEPARAMETERS_H

#include <QJsonObject>

class SliceParameters
{
public:
    SliceParameters();

public:
    double layerHeight;
    int nozzleTemp;
    int bedTemp;
    int wallLoops;
    int infillDensity;
    bool supportsEnabled;

    // more....? gets built from Printer, Material, and Process Profiles

    QJsonObject toJson() const;
};

#endif // SLICEPARAMETERS_H
