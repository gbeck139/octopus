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

    void runSlice(const SliceParameters& params);
    QProcess* getProcess() const { return process; }

signals:
    void sliceStarted();
    void sliceFinished(QString gcodePath);
    void sliceFailed(QString error);
    void slicerLog(QString log);

private slots:
    void onFinished(int exitCode, QProcess::ExitStatus status);
    void readStdOut();
    void readStdErr();

private:
    QProcess* process = nullptr;
    QString outputPath;
};

#endif
