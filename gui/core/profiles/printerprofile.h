#ifndef PRINTERPROFILE_H
#define PRINTERPROFILE_H

#include <QString>
#include <QJsonObject>

class PrinterProfile
{
public:
    explicit PrinterProfile(const QString& profileId);

    //JSON
    static PrinterProfile fromJson(const QJsonObject& obj);
    QJsonObject toJson() const;

    // Get ID
    QString getId() const;
    QString getDisplayName() const;

    // Get Limits
    int getMaxNozzleTemp() const;
    int getMaxBedTemp() const;

    double getBuildX() const;
    double getBuildY() const;
    double getBuildZ() const;

    // Setters
    void setDisplayName(const QString& name);
    void setMaxNozzleTemp(int temp);
    void setMaxBedTemp(int temp);
    void setBuildVolume(double x, double y, double z);

private:
    QString id;
    QString displayName;

    int maxNozzleTemp;
    int maxBedTemp;

    double buildX;
    double buildY;
    double buildZ;
};

#endif // PRINTERPROFILE_H
