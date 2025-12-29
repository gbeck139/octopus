#include "printerprofile.h"


PrinterProfile::PrinterProfile(const QString &profileId)
    : id(profileId),
    displayName(id)
{
}

PrinterProfile PrinterProfile::fromJson(const QJsonObject &obj)
{
    PrinterProfile p(obj["id"].toString());

    p.displayName = obj["name"].toString();
    p.maxNozzleTemp = obj["limits"]["nozzleMax"].toInt();
    p.maxBedTemp = obj["limits"]["bedMax"].toInt();

    auto build = obj["buildVolume"].toObject();
    p.buildX = build["x"].toDouble();
    p.buildY = build["y"].toDouble();
    p.buildZ = build["z"].toDouble();

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

void PrinterProfile::setDisplayName(const QString &name)
{
    displayName = name;
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

