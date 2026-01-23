#ifndef SLICEPARAMETERS_H
#define SLICEPARAMETERS_H

#include <QJsonObject>

struct SliceParameters
{
    double layerHeight = 0.2;
    int nozzleTemp = 200;
    int bedTemp = 60;
    int wallLoops = 2;
    int infillDensity = 20;
    bool supportsEnabled = false;

    QJsonObject toJson() const
    {
        QJsonObject obj;
        obj["layer_height"] = layerHeight;
        obj["nozzle_temp"] = nozzleTemp;
        obj["bed_temp"] = bedTemp;
        obj["wall_loops"] = wallLoops;
        obj["infill_density"] = infillDensity;
        obj["supports"] = supportsEnabled;
        return obj;
    }
};

#endif // SLICEPARAMETERS_H
