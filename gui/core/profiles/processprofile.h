#ifndef PROCESSPROFILE_H
#define PROCESSPROFILE_H

#include <QString>
#include <QJsonObject>

class ProcessProfile
{
public:
    explicit ProcessProfile(const QString& id, bool system);
    static ProcessProfile* fromJson(const QJsonObject& obj, bool system);
    QJsonObject toJson() const;

    QString getId() const;
    QString getDisplayName() const;
    bool isSystemProfile() const;

    // Quality
    double getLayerHeight() const;

    // Strength
    int getWallLoops() const;
    int getInfillDensity() const;

    // Support
    bool supportsEnabled() const;

    // Setters
    void setId(const QString& id);
    void setDisplayName(const QString& name);
    void setLayerHeight(double h);
    void setWallLoops(int w);
    void setInfillDensity(int d);
    void setSupportsEnabled(bool e);
    void setIsSystem(bool system);

    ProcessProfile* clone() const;

private:
    QString id;
    QString displayName;
    double layerHeight = 0.2;
    int wallLoops = 2;
    int infillDensity = 20;
    bool supports = false;
    bool isSystem;
};

#endif // PROCESSPROFILE_H
