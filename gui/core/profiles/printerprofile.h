#ifndef PRINTERPROFILE_H
#define PRINTERPROFILE_H

#include <QString>
#include <QJsonObject>

class PrinterProfile
{
public:
    explicit PrinterProfile(const QString& profileId, bool system);

    //JSON
    static PrinterProfile* fromJson(const QJsonObject& obj, bool system);
    QJsonObject toJson() const;

    // ID
    QString getId() const;
    QString getDisplayName() const;
    bool isSystemProfile() const;

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

    PrinterProfile* clone() const;

private:
    QString id;
    QString displayName;

    int maxNozzleTemp = 0;
    int maxBedTemp = 0;

    double buildX = 0;
    double buildY = 0;
    double buildZ = 0;

    bool isSystem;
};

#endif // PRINTERPROFILE_H
