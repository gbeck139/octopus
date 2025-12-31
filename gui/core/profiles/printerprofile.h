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

    double getNozzleDiameter() const;
    void setNozzleDiameter(double d);

    double getXMin() const;
    double getXMax() const;
    double getYMin() const;
    double getYMax() const;
    double getZMin() const;
    double getZMax() const;
    double getRotMin() const;
    double getRotMax() const;

    void setAxisLimits(double _xMin, double _xMax,
                       double _yMin, double _yMax,
                       double _zMin, double _zMax,
                       double _rotMin, double _rotMax);

    // Setters
    void setId(QString& name);
    void setDisplayName(const QString& name);
    void setIsSystem(bool system);
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

    double nozzleDiameter = 0.4;

    // Axis limits
    double xMin = 0;
    double xMax = 0;
    double yMin = 0;
    double yMax = 0;
    double zMin = 0;
    double zMax = 0;
    double rotMin = 0;
    double rotMax = 0;

    bool isSystem;
};

#endif // PRINTERPROFILE_H
