#include "appconfig.h"
#include <QDialog>

AppConfig::AppConfig(QObject *parent)
    : QObject{parent},
    appSettings("Research", "SlicerApp")
{}

bool AppConfig::isFirstRun() const
{
    return !appSettings.value("app/firstRunCompleted", false).toBool();
}

void AppConfig::setFirstRunCompleted(bool completed)
{
    appSettings.setValue("app/firstRunCompleted", completed);
}

void AppConfig::setActivePrinter(QString printerId)
{

    if (printerId == appSettings.value("profile/activePrinter"))
        return;

    appSettings.setValue("profile/activePrinter", printerId);
    qDebug() << "[APPCONFIG] Saved current printer profile to: " << appSettings.value("profile/activePrinter");

    emit activePrinterChanged(printerId);

}

void AppConfig::setActiveMaterial(QString materialId)
{
    appSettings.setValue("profile/activeMaterial", materialId);
    qDebug() << "[APPCONFIG] Saved current material profile to: " << materialId;
}

void AppConfig::setActiveProcess(QString processId)
{
    appSettings.setValue("profile/activeProcess", processId);
    qDebug() << "[APPCONFIG] Saved current process profile to: " << processId;
    emit activePrinterChanged(processId);
}

QString AppConfig::getActivePrinter() const
{
    qDebug() << "[APPCONFIG] Current active printer:" << appSettings.value("profile/activePrinter").toString();
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

