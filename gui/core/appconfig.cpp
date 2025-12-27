#include "appconfig.h"
#include <QDialog>

AppConfig::AppConfig(QObject *parent)
    : QObject{parent},
    appSettings("Research", "SlicerApp")
{}

bool AppConfig::isFirstRun() const
{
    // If the key does not exist, it's the first run
    return !appSettings.contains("app/firstRunCompleted");
}

void AppConfig::setFirstRunCompleted()
{
    //if (result == QDialog::Accepted) {
    appSettings.setValue("app/firstRunCompleted", true);
    //}
}

void AppConfig::setDefaultPrinter(int printerType)
{
    appSettings.setValue("app/defaultPrinter", printerType);
    qDebug() << "Set current printer to: " << printerType;
}
