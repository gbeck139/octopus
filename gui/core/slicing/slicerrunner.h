#ifndef SLICERRUNNER_H
#define SLICERRUNNER_H

#include <QObject>
#include <QProcess>
#include "sliceparameters.h"

class SlicerRunner : public QObject
{
    Q_OBJECT
public:
    explicit SlicerRunner(QObject *parent = nullptr);

    void runSlice(const QString& stlPath, const SliceParameters& params);

signals:
    void sliceFinished(QString gcodePath); /////////////
    void sliceFailed(QString error); //////////////

private slots:
    void onProcessFinished(int exitCode);

private:
    QProcess process;
};

#endif // SLICERRUNNER_H
