#include "slicerrunner.h"

#include <QCoreApplication>
#include <QFile>
#include <QFileInfo>
#include <QDebug>

SlicerRunner::SlicerRunner(QObject *parent)
    : QObject(parent)
{
}

void SlicerRunner::runSlice(const SliceParameters& params)
{
    qDebug() << "SlicerRunner::runSlice CALLED";

    QStringList errors;
    if (!params.validate(errors))
    {
        qDebug() << "PARAM VALIDATION FAILED:";
        for (const auto& e : errors)
            qDebug() << "   -" << e;

        emit sliceFailed(errors.join("\n"));
        return;
    }

    qDebug() << "Parameters validated OK";

    emit sliceStarted();

    QString base =
        QCoreApplication::applicationDirPath();

    qDebug() << "App base path:" << base;

    QString exe =
        base + "/slicerbundle/config_slicer_pipeline.exe";

     qDebug() << "Slicer exe path:" << exe;

    if (!QFile::exists(exe))
    {
        qDebug() << "SLICER EXE NOT FOUND";
        emit sliceFailed("Missing slicer exe:\n" + exe);
        return;
    }

    qDebug() << "Slicer exe exists";

    QString configPath =
        base + "/slicerbundle/example_config.json";

    qDebug() << "Writing config to:" << configPath;

    if (!params.saveToFile(configPath))
    {
        qDebug() << "FAILED TO WRITE CONFIG JSON";
        emit sliceFailed("Failed writing config JSON.");
        return;
    }

    qDebug() << "Config JSON written successfully";

    outputPath =
        base + "/slicerbundle/output_gcode/"
        + params.modelName
        + "_reformed.gcode";

    qDebug() << "Expected output path:" << outputPath;

    process = new QProcess(this);

    //connect(process, &QProcess::readyReadStandardOutput,
    //        this, &SlicerRunner::readStdOut);

    //connect(process, &QProcess::readyReadStandardError,
    //        this, &SlicerRunner::readStdErr);

    connect(process, &QProcess::readyReadStandardOutput, this, [this]() {
        qDebug().noquote() << process->readAllStandardOutput();
    });

    connect(process, &QProcess::readyReadStandardError, this, [this]() {
        qDebug().noquote() << process->readAllStandardError();
    });

    connect(process,
            QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
            this,
            &SlicerRunner::onFinished);

    QStringList args;
    args << "--stl" << params.stlPath
         << "--model" << params.modelName
         << "--prusa" << params.prusaSlicerPath
         << "--config" << configPath;

    qDebug() << "Launching process:";
    qDebug() << "EXE:" << exe;
    qDebug() << "ARGS:" << args;

    QString workDir =
        QCoreApplication::applicationDirPath();

    qDebug() << "📂 Setting working directory:" << workDir;

    process->setWorkingDirectory(workDir);

    process->start(exe, args);

    qDebug() << "process->start() called";

    if (!process->waitForStarted(3000))
    {
        qDebug() << "PROCESS FAILED TO START";
        emit sliceFailed("Process failed to start.");
        return;
    }

    qDebug() << "PROCESS IS RUNNING";
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
