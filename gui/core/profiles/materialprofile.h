#ifndef MATERIALPROFILE_H
#define MATERIALPROFILE_H

#include <QString>

class MaterialProfile
{
public:
    MaterialProfile(QString profileId);

    QString getId() const;

    int getNozzleTemp() const;
    void setNozzleTemp(int temp);

    int getNozzleMin() const;
    int getNozzleMax() const;

private:
    QString id;

    int nozzleMin;
    int nozzleMax;
    int nozzleDefault;

};

#endif // MATERIALPROFILE_H
