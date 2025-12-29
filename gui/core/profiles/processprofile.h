#ifndef PROCESSPROFILE_H
#define PROCESSPROFILE_H

#include <QString>

class ProcessProfile
{
public:
    ProcessProfile(QString profileID);

    QString getId() const;

    double getLayerHeight() const;
    void setLayerHeight(double h);

private:
    QString id;
    double layerHeight;
};

#endif // PROCESSPROFILE_H
