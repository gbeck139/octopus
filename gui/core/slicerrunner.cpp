#include "slicerrunner.h"
#include <QJsonDocument>
#include <QTemporaryFile>
#include <QDebug>

SlicerRunner::SlicerRunner(QObject *parent)
    : QObject{parent}
{
    connect(&process, &QProcess::finished,
            this, &SlicerRunner::onProcessFinished);
}

void SlicerRunner::runSlice(const QString& stlPath, const SliceParameters& params)
{
    //TODO:

    qDebug() << "[SLICER RUNNER] Starting slice for:" << stlPath;

    // QTemporaryFile jsonFile;
    // jsonFile.open();

    // QJsonDocument doc(params.toJson());
    // jsonFile.write(doc.toJson());
    // jsonFile.close();

    // qDebug() << "[SLICER RUNNER] Launching Python with params JSON";

    // process.start("python", QStringList() << "slicer.py" << stlPath << jsonFile.fileName());

    //
}

void SlicerRunner::onProcessFinished(int exitCode)
{
    qDebug() << "[SLICER RUNNER] Process finished, code:" << exitCode;

    if (exitCode == 0)
        emit sliceFinished("output.gcode");
    else
        emit sliceFailed(process.readAllStandardError());
}
