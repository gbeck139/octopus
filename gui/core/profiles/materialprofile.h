#ifndef MATERIALPROFILE_H
#define MATERIALPROFILE_H

#include <QString>
#include <QJsonObject>

class MaterialProfile
{
public:
    explicit MaterialProfile(const QString& profileId);

    static MaterialProfile* fromJson(const QJsonObject& obj, bool system);
    QJsonObject toJson() const;

    QString getId() const;
    QString getDisplayName() const;

    int getNozzleTemp() const;
    void setNozzleTemp(int temp);

    int getBedTemp() const;
    void setBedTemp(int temp);

    int getNozzleMin() const;
    int getNozzleMax() const;

    MaterialProfile* clone() const;

private:
    QString id;
    QString displayName;

    int nozzleMin;
    int nozzleMax;
    int nozzleDefault;

    int nozzleTemp;
    int bedTemp;

};

#endif // MATERIALPROFILE_H
