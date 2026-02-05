#include "printerprofile.h"


PrinterProfile::PrinterProfile(const QString &profileId, bool system)
    : id(profileId),
    displayName(profileId),
    isSystem(system)
{
}

PrinterProfile* PrinterProfile::fromJson(const QJsonObject &obj, bool system)
{
    auto* p = new PrinterProfile(obj["id"].toString(), system);

    p->displayName = obj["name"].toString();

    auto limits = obj["limits"].toObject();
    p->maxNozzleTemp = limits["nozzleMax"].toInt();
    p->maxBedTemp = limits["bedMax"].toInt();

    auto build = obj["buildVolume"].toObject();
    p->buildX = build["x"].toDouble();
    p->buildY = build["y"].toDouble();
    p->buildZ = build["z"].toDouble();

    p->nozzleDiameter = obj["nozzleDiameter"].toDouble(0.4);

    auto axis = obj["axisLimits"].toObject();
    p->xMin = axis["xMin"].toDouble();
    p->xMax = axis["xMax"].toDouble();
    p->yMin = axis["yMin"].toDouble();
    p->yMax = axis["yMax"].toDouble();
    p->zMin = axis["zMin"].toDouble();
    p->zMax = axis["zMax"].toDouble();
    p->rotMin = axis["rotMin"].toDouble();
    p->rotMax = axis["rotMax"].toDouble();

    qDebug() << "[PRINTERPROFILE] Loaded PrinterProfile:" << p->displayName << "from JSON";

    return p;
}

QJsonObject PrinterProfile::toJson() const
{
    QJsonObject obj;
    obj["id"] = id;
    obj["name"] = displayName;

    QJsonObject limits;
    limits["nozzleMax"] = maxNozzleTemp;
    limits["bedMax"] = maxBedTemp;
    obj["limits"] = limits;

    QJsonObject build;
    build["x"] = buildX;
    build["y"] = buildY;
    build["z"] = buildZ;
    obj["buildVolume"] = build;

    obj["nozzleDiameter"] = nozzleDiameter;

    QJsonObject axis;
    axis["xMin"] = xMin;
    axis["xMax"] = xMax;
    axis["yMin"] = yMin;
    axis["yMax"] = yMax;
    axis["zMin"] = zMin;
    axis["zMax"] = zMax;
    axis["rotMin"] = rotMin;
    axis["rotMax"] = rotMax;
    obj["axisLimits"] = axis;

    qDebug() << "[PRINTERPROFILE] Created JSON from PrinterProfile:" << this->getDisplayName();

    return obj;
}

QString PrinterProfile::getId() const
{
    return id;
}

QString PrinterProfile::getDisplayName() const
{
    return displayName;
}

bool PrinterProfile::isSystemProfile() const
{
    return isSystem;
}

int PrinterProfile::getMaxNozzleTemp() const
{
    return maxNozzleTemp;
}

int PrinterProfile::getMaxBedTemp() const
{
    return maxBedTemp;
}

double PrinterProfile::getBuildX() const
{
    return buildX;
}

double PrinterProfile::getBuildY() const
{
    return buildY;
}

double PrinterProfile::getBuildZ() const
{
    return buildZ;
}

double PrinterProfile::getNozzleDiameter() const { return nozzleDiameter; }
void PrinterProfile::setNozzleDiameter(double d) { nozzleDiameter = d; }

double PrinterProfile::getXMin() const { return xMin; }
double PrinterProfile::getXMax() const { return xMax; }
double PrinterProfile::getYMin() const { return yMin; }
double PrinterProfile::getYMax() const { return yMax; }
double PrinterProfile::getZMin() const { return zMin; }
double PrinterProfile::getZMax() const { return zMax; }
double PrinterProfile::getRotMin() const { return rotMin; }
double PrinterProfile::getRotMax() const { return rotMax; }

void PrinterProfile::setAxisLimits(double _xMin, double _xMax,
                   double _yMin, double _yMax,
                   double _zMin, double _zMax,
                   double _rotMin, double _rotMax)
{
    xMin = _xMin; xMax = _xMax;
    yMin = _yMin; yMax = _yMax;
    zMin = _zMin; zMax = _zMax;
    rotMin = _rotMin; rotMax = _rotMax;
}

void PrinterProfile::setId(QString &name)
{
    id = name;
}

void PrinterProfile::setDisplayName(const QString &name)
{
    displayName = name;
}

void PrinterProfile::setIsSystem(bool system)
{
    isSystem = system;
}

void PrinterProfile::setMaxNozzleTemp(int temp)
{
    maxNozzleTemp = temp;
}

void PrinterProfile::setMaxBedTemp(int temp)
{
    maxBedTemp = temp;
}

void PrinterProfile::setBuildVolume(double x, double y, double z)
{
    buildX = x;
    buildY = y;
    buildZ = z;
}

PrinterProfile *PrinterProfile::clone() const
{
    qDebug() << "[PRINTERPROFILE] Cloning Printer profile:" << this->getDisplayName();
    auto* copy = new PrinterProfile(id, isSystem);
    *copy = *this;
    return copy;
}

