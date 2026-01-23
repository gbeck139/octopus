#ifndef SLICEPARAMETERS_H
#define SLICEPARAMETERS_H

#include <QJsonObject>

class SliceParameters
{
public:
    SliceParameters();

public:

    ///
    /// Immutable slicing parameter bundle.
    /// Built from Drafts / Profiles at slice time.
    ///
    // struct SliceParameters
    // {
    //     double layerHeight = 0.0;
    //     int nozzleTemp = 0;
    //     int bedTemp = 0;
    //     int wallLoops = 0;
    //     int infillDensity = 0;
    //     bool supportsEnabled = false;

    //     QJsonObject toJson() const;
    // };


    double layerHeight;
    int nozzleTemp;
    int bedTemp;
    int wallLoops;
    int infillDensity;
    bool supportsEnabled;

    // more...

    QJsonObject toJson() const;
};

#endif // SLICEPARAMETERS_H
