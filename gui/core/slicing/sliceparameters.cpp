#include "sliceparameters.h"

#include <QFile>

QString SliceParameters::slicingModeToString() const
{
    switch (slicingMode)
    {
    case SlicingMode::Planar:
        return "planar";

    case SlicingMode::RadialNonPlanar:
        return "radial_non_planar";
    }

    return "radial_non_planar";
}

bool SliceParameters::validate(QStringList& errors) const
{
    errors.clear();

    // =========================
    // PRINT VALIDATION
    // =========================

    if (layerHeight <= 0.0 || layerHeight > 1.0)
    {
        errors.append("Layer height must be between 0 and 1 mm.");
    }

    if (nozzleTemperature < 0 || nozzleTemperature > 400)
    {
        errors.append("Nozzle temperature is invalid.");
    }

    if (bedTemperature < 0 || bedTemperature > 150)
    {
        errors.append("Bed temperature is invalid.");
    }

    if (infillDensity < 0 || infillDensity > 100)
    {
        errors.append("Infill density must be between 0 and 100.");
    }

    if (wallLoops < 1 || wallLoops > 20)
    {
        errors.append("Wall loops must be between 1 and 20.");
    }

    // =========================
    // DEFORMATION VALIDATION
    // =========================

    if (angleBase < 0 || angleBase > 90)
    {
        errors.append("Angle base must be between 0 and 90.");
    }

    if (angleFactor < 0 || angleFactor > 180)
    {
        errors.append("Angle factor must be between 0 and 180.");
    }

    // =========================
    // FILE VALIDATION
    // =========================

    if (stlPath.isEmpty())
    {
        errors.append("No STL file selected.");
    }

    if (prusaSlicerPath.isEmpty())
    {
        errors.append("PrusaSlicer path is missing.");
    }

    return errors.isEmpty();
}

QJsonObject SliceParameters::toJson() const
{
    QJsonObject root;

    // =========================
    // MODEL
    // =========================

    QJsonObject model;
    model["rotX"] = rotX;
    model["rotY"] = rotY;
    model["rotZ"] = rotZ;

    root["model"] = model;

    // =========================
    // DEFORMATION
    // =========================

    QJsonObject deformation;
    deformation["angle_base"] = angleBase;
    deformation["angle_factor"] = angleFactor;

    root["deformation"] = deformation;

    // =========================
    // PRINT
    // =========================

    QJsonObject print;
    print["layer_height"] = layerHeight;
    print["nozzle_temperature"] = nozzleTemperature;
    print["bed_temperature"] = bedTemperature;
    print["infill_density"] = infillDensity;
    print["wall_loops"] = wallLoops;
    print["supports_enabled"] = supportsEnabled;

    root["print"] = print;

    // =========================
    // PIPELINE
    // =========================

    root["slicing_mode"] = slicingModeToString();

    return root;
}

bool SliceParameters::saveToFile(const QString& filePath) const
{
    QFile file(filePath);

    if (!file.open(QIODevice::WriteOnly))
    {
        return false;
    }

    QJsonDocument doc(toJson());

    file.write(doc.toJson(QJsonDocument::Indented));

    file.close();

    return true;
}

SliceParameters SliceParameters::fromJson(const QJsonObject& obj)
{
    SliceParameters params;

    // Future implementation

    return params;
}
