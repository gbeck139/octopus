#include "profilemanager.h"

#include <QDir>
#include <QFile>
#include <QJsonDocument>
#include <QCoreApplication>
#include <QRegularExpression>

ProfileManager::ProfileManager(QObject *parent)
    : QObject{parent}
{
    loadPrinterProfiles();
}

QList<PrinterViewData> ProfileManager::getSystemPrintersForView() const
{
    QList<PrinterViewData> printerData;
    for (const PrinterProfile* profile : systemPrinters) {
        printerData.append(makeViewData(*profile));
    }
    return printerData;
}

QList<PrinterViewData> ProfileManager::getUserPrintersForView() const
{
    QList<PrinterViewData> printerData;
    for (const PrinterProfile* profile : userPrinters) {
        printerData.append(makeViewData(*profile));
    }
    return printerData;
}

QString ProfileManager::getActivePrinter()
{
    return activePrinterId;
}

PrinterViewData ProfileManager::getActivePrinterDataForView()
{
    if (systemPrinters.contains(activePrinterId)) {
        return makeViewData(*systemPrinters.value(activePrinterId));
    }

    if (userPrinters.contains(activePrinterId)) {
        return makeViewData(*userPrinters.value(activePrinterId));
    }

    qDebug() << "[PROFILE MANAGER] Active printer not found:" << activePrinterId;
    return {};
}

void ProfileManager::setActivePrinter(const QString &printerId)
{
    if (!userPrinters.contains(printerId) && !systemPrinters.contains(printerId)) {
        qDebug() << "[PROFILE MANAGER] [ERROR] Unknown printer:" << printerId;
        return;
    }

    if (printerId == activePrinterId) {
        return;
    }

    qDebug() << "[PROFILE MANAGER] Saved current printer";

    activePrinterId = printerId;
    emit activePrinterChanged(printerId);
}

void ProfileManager::addUserPrinter(const PrinterProfile& profile)
{
    PrinterProfile* copy = profile.clone();
    copy->setIsSystem(false);

    // Ensure unique ID
    int suffix = 0;
    QString newId = generateUniquePrinterId(copy->getId(), &suffix);
    copy->setId(newId);

    // Update display name: add suffix if necessary
    if (suffix > 0) {
        copy->setDisplayName(
            copy->getDisplayName() + " (" + QString::number(suffix) + ")");
    }

    userPrinters.insert(newId, copy);

    // Write JSON to user directory
    savePrinterProfile(copy);

    activePrinterId = newId;

    emit printersChanged();
    emit activePrinterChanged(newId);
}

void ProfileManager::updateUserPrinter(const PrinterProfile &profile)
{
    const QString id = profile.getId();

    // Ensure unique ID
    if (!userPrinters.contains(id)) {
        qWarning() << "[PROFILE MANAGER] Tried to update non-user printer:" << id;
        return;
    }

    delete userPrinters[id];
    userPrinters[id] = profile.clone();

    // Overwrite JSON
    savePrinterProfile(&profile);

    if (id == activePrinterId) {
        emit activePrinterChanged(id);
    }
}

void ProfileManager::savePrinterProfile(const PrinterProfile *profile)
{
    QString path = getUserPrinterDir() + "/" + profile->getId() + ".json";

    QFile file(path);
    if (!file.open(QIODevice::WriteOnly))
        return;

    QJsonDocument doc(profile->toJson());
    file.write(doc.toJson(QJsonDocument::Indented));
    file.close();
}

void ProfileManager::deleteUserPrinter(const QString &printerId)
{
    if (!userPrinters.contains(printerId)) {
        qDebug() << "[PROFILE MANAGER] Tried to delete non-existent printer:" << printerId;
        return;
    }

    // Delete JSON file
    QString path = getUserPrinterDir() + "/" + printerId + ".json";
    if (QFile::exists(path)) {
        QFile::remove(path);
    }

    // Remove from map
    userPrinters.remove(printerId);

    emit printersChanged();

    // If the deleted printer was active, set a new active printer
    if (activePrinterId == printerId) {
        if (!userPrinters.isEmpty()) {
            setActivePrinter(userPrinters.firstKey());
        } else if (!systemPrinters.isEmpty()) {
            setActivePrinter(systemPrinters.firstKey());
        } else {
            activePrinterId.clear();
        }
    }
}

void ProfileManager::loadPrinterProfiles()
{
    qDeleteAll(systemPrinters);
    qDeleteAll(userPrinters);

    systemPrinters.clear();
    userPrinters.clear();

    loadPrinterDirectory(getSystemPrinterDir(), true);
    loadPrinterDirectory(getUserPrinterDir(), false);

    qDebug() << "[PROFILE MANAGER] Loaded printers:" << systemPrinters.keys() << userPrinters.keys() << ", from:" << getSystemPrinterDir() << "and" << getUserPrinterDir();
}

void ProfileManager::loadPrinterDirectory(const QString &path, bool system)
{
    QStringList files;

    if (system) {
        // For resources, QDir can list files inside the resource
        QDir resDir(path);
        files = resDir.entryList(QStringList() << "*.json", QDir::Files);
    } else {
        // For user folder on disk
        QDir dir(path);
        if (!dir.exists()) {
            dir.mkpath(".");
        }
        files = dir.entryList(QStringList() << "*.json", QDir::Files);
    }

    for (const QString &fileName : files) {
        QString fullPath = path + "/" + fileName;
        QFile file(fullPath);
        if (!file.open(QIODevice::ReadOnly)) {
            qDebug() << "[PROFILE MANAGER] [ERROR] Failed to open printer profile:" << fullPath;
            continue;
        }

        QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
        if (!doc.isObject())
            continue;

        PrinterProfile* profile = PrinterProfile::fromJson(doc.object(), system);
        (system ? systemPrinters : userPrinters).insert(profile->getId(), profile);

        file.close();

        qDebug() << "[PROFILE MANAGER] System printers loaded:" << systemPrinters.keys();
        qDebug() << "[PROFILE MANAGER] User printers loaded:" << userPrinters.keys();
    }
}

QString ProfileManager::getSystemPrinterDir() const
{
    return ":/json/json/";
}

QString ProfileManager::getUserPrinterDir() const
{
    return QCoreApplication::applicationDirPath() + "/core/profiles/printers/user";
}

QString ProfileManager::generateUniquePrinterId(const QString &baseId, int *outSuffix) const
{
    qDebug() << "[PROFILE MANAGER] Generating a unique ID!";
    if (!systemPrinters.contains(baseId) && !userPrinters.contains(baseId)) {
        if (outSuffix) *outSuffix = 0;
        return baseId;
    }

    int suffix = 1;
    QString newId;
    do {
        newId = baseId + QString::number(suffix++);
    } while (systemPrinters.contains(newId) || userPrinters.contains(newId));

    if (outSuffix) *outSuffix = suffix - 1;
    return newId;
}

PrinterViewData ProfileManager::makeViewData(const PrinterProfile &profile) const
{
    PrinterViewData view;
    view.id = profile.getId();
    view.name = profile.getDisplayName();
    view.isSystem = profile.isSystemProfile();
    return view;
}

