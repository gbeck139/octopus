#include "slicerrunner.h"

#include <QCoreApplication>
#include <QFile>
#include <QFileInfo>

SlicerRunner::SlicerRunner(QObject *parent)
    : QObject(parent)
{
}

void SlicerRunner::runSlice(const SliceParameters& params)
{
    QStringList errors;
    if (!params.validate(errors))
    {
        emit sliceFailed(errors.join("\n"));
        return;
    }

    emit sliceStarted();

    QString base =
        QCoreApplication::applicationDirPath();

    QString exe =
        base + "/slicerbundle/config_slicer_pipeline.exe";

    if (!QFile::exists(exe))
    {
        emit sliceFailed("Missing slicer exe:\n" + exe);
        return;
    }

    QString configPath =
        base + "/slicerbundle/example_config.json";

    if (!params.saveToFile(configPath))
    {
        emit sliceFailed("Failed writing config JSON.");
        return;
    }

    outputPath =
        base + "/slicerbundle/output_gcode/"
        + params.modelName
        + "_reformed.gcode";

    process = new QProcess(this);

    connect(process, &QProcess::readyReadStandardOutput,
            this, &SlicerRunner::readStdOut);

    connect(process, &QProcess::readyReadStandardError,
            this, &SlicerRunner::readStdErr);

    connect(process,
            QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            this,
            &SlicerRunner::onFinished);

    QStringList args;
    args << "--stl" << params.stlPath
         << "--model" << params.modelName
         << "--prusa" << params.prusaSlicerPath
         << "--config" << configPath;

    process->start(exe, args);
}

void SlicerRunner::readStdOut()
{
    emit slicerLog(process->readAllStandardOutput());
}

void SlicerRunner::readStdErr()
{
    emit slicerLog(process->readAllStandardError());
}

void SlicerRunner::onFinished(int exitCode, QProcess::ExitStatus status)
{
    if (status == QProcess::NormalExit && exitCode == 0)
        emit sliceFinished(outputPath);
    else
        emit sliceFailed(process->readAllStandardError());

    process->deleteLater();
    process = nullptr;
}
