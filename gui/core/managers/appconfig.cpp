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

void AppConfig::markFirstRunCompleted()
{
    appSettings.setValue("app/firstRunCompleted", true);
}

QString AppConfig::getActivePrinterId() const
{
    qDebug() << "[APPCONFIG] Current active printer:" << appSettings.value("profile/activePrinter").toString();
    return appSettings.value("profile/activePrinter").toString();
}

QString AppConfig::getActiveMaterialId() const
{
    return appSettings.value("profile/activeMaterial").toString();
}

QString AppConfig::getActiveProcessId() const
{
    return appSettings.value("profile/activeProcess").toString();
}

void AppConfig::setActivePrinterId(const QString& printerId)
{

    if (printerId == appSettings.value("profile/activePrinter"))
        return;

    appSettings.setValue("profile/activePrinter", printerId);
    qDebug() << "[APPCONFIG] Saved current printer profile to: " << printerId;

    emit activePrinterChanged(printerId);

}

void AppConfig::setActiveMaterialId(const QString& materialId)
{
    if (materialId == appSettings.value("profile/activeMaterial"))
        return;

    appSettings.setValue("profile/activeMaterial", materialId);
    qDebug() << "[APPCONFIG] Saved current material profile to: " << materialId;

    emit activeMaterialChanged(materialId);
}

void AppConfig::setActiveProcessId(const QString& processId)
{
    if (processId == appSettings.value("profile/activeProcess"))
        return;

    appSettings.setValue("profile/activeProcess", processId);
    qDebug() << "[APPCONFIG] Saved current process profile to: " << processId;

    emit activeProcessChanged(processId);
}
