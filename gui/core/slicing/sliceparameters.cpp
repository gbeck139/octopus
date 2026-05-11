#include "sliceparameters.h"

#include <QFile>
#include <QJsonDocument>

bool SliceParameters::validate(QStringList& errors) const
{
    errors.clear();

    if (stlPath.isEmpty())
        errors << "Missing STL path.";

    if (modelName.isEmpty())
        errors << "Missing model name.";

    if (prusaSlicerPath.isEmpty())
        errors << "Missing PrusaSlicer path.";

    if (layerHeight <= 0.0 || layerHeight > 1.0)
        errors << "Layer height must be between 0 and 1.";

    if (nozzleTemperature < 0 || nozzleTemperature > 500)
        errors << "Invalid nozzle temperature.";

    if (bedTemperature < 0 || bedTemperature > 200)
        errors << "Invalid bed temperature.";

    if (infillDensity < 0 || infillDensity > 100)
        errors << "Infill must be 0–100.";

    if (wallLoops < 1 || wallLoops > 20)
        errors << "Wall loops must be 1–20.";

    if (angleBase < 0 || angleBase > 90)
        errors << "Angle base must be 0–90.";

    if (angleFactor < 0 || angleFactor > 180)
        errors << "Angle factor must be 0–180.";

    return errors.isEmpty();
}

QJsonObject SliceParameters::toJson() const
{
    QJsonObject root;

    QJsonObject model;
    model["rotX"] = rotX;
    model["rotY"] = rotY;
    model["rotZ"] = rotZ;
    model["modelName"] = modelName;

    QJsonObject deformation;
    deformation["angle_base"] = angleBase;
    deformation["angle_factor"] = angleFactor;

    QJsonObject print;
    print["layer_height"] = layerHeight;
    print["nozzle_temperature"] = nozzleTemperature;
    print["bed_temperature"] = bedTemperature;
    print["infill_density"] = infillDensity;
    print["wall_loops"] = wallLoops;
    print["supports_enabled"] = supportsEnabled;

    QJsonObject paths;
    paths["stl"] = stlPath;
    paths["prusa"] = prusaSlicerPath;
    paths["output_dir"] = outputDirectory;

    root["model"] = model;
    root["deformation"] = deformation;
    root["print"] = print;
    root["paths"] = paths;

    root["slicing_mode"] =
        (slicingMode == SlicingMode::Planar)
            ? "planar"
            : "radial_non_planar";

    return root;
}

bool SliceParameters::saveToFile(const QString& filePath) const
{
    QFile file(filePath);

    if (!file.open(QIODevice::WriteOnly))
        return false;

    QJsonDocument doc(toJson());

    file.write(doc.toJson(QJsonDocument::Indented));

    return true;
}

SliceParameters SliceParameters::fromJson(const QJsonObject& obj)
{
    SliceParameters p;

    auto model = obj["model"].toObject();
    auto print = obj["print"].toObject();
    auto deform = obj["deformation"].toObject();
    auto paths = obj["paths"].toObject();

    p.rotX = model["rotX"].toDouble();
    p.rotY = model["rotY"].toDouble();
    p.rotZ = model["rotZ"].toDouble();
    p.modelName = model["modelName"].toString();

    p.angleBase = deform["angle_base"].toDouble();
    p.angleFactor = deform["angle_factor"].toDouble();

    p.layerHeight = print["layer_height"].toDouble();
    p.nozzleTemperature = print["nozzle_temperature"].toInt();
    p.bedTemperature = print["bed_temperature"].toInt();
    p.infillDensity = print["infill_density"].toInt();
    p.wallLoops = print["wall_loops"].toInt();
    p.supportsEnabled = print["supports_enabled"].toBool();

    p.stlPath = paths["stl"].toString();
    p.prusaSlicerPath = paths["prusa"].toString();
    p.outputDirectory = paths["output_dir"].toString();

    return p;
}
