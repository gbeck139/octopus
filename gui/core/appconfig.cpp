#include "appconfig.h"
#include <QDialog>

AppConfig::AppConfig(QObject *parent)
    : QObject{parent},
    appSettings("Research", "SlicerApp")
{}

bool AppConfig::isFirstRun() const
{
    //return !appSettings.contains("app/firstRunCompleted");
    return !appSettings.value("app/firstRunCompleted", false).toBool();
}

void AppConfig::setFirstRunCompleted(bool completed)
{
    appSettings.setValue("app/firstRunCompleted", completed);
}

void AppConfig::setActivePrinter(QString printerId)
{
    appSettings.setValue("profile/activePrinter", printerId);
    qDebug() << "[UPDATE] Saved current printer profile to: " << printerId;
}

void AppConfig::setActiveMaterial(QString materialId)
{
    appSettings.setValue("profile/activeMaterial", materialId);
    qDebug() << "[UPDATE] Saved current material profile to: " << materialId;
}

void AppConfig::setActiveProcess(QString processId)
{
    appSettings.setValue("profile/activeProcess", processId);
    qDebug() << "[UPDATE] Saved current process profile to: " << processId;
}

QString AppConfig::getActivePrinter() const
{
    return appSettings.value("profile/activePrinter").toString();
}

QString AppConfig::getActiveMaterial() const
{
    return appSettings.value("profile/activeMaterial").toString();
}

QString AppConfig::getActiveProcess() const
{
    return appSettings.value("profile/activeProcess").toString();
}

