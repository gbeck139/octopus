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

PrinterProfile *PrinterProfile::clone() const
{
    qDebug() << "[PRINTERPROFILE] Cloning Printer profile:" << this->getDisplayName();
    auto* copy = new PrinterProfile(id, isSystem);
    *copy = *this;
    return copy;
}

