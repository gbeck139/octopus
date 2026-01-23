#include "profilemanager.h"

#include <QDir>
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
    QList<PrinterViewData> profiles;
    for (PrinterProfile* profile : systemPrinters.values()) {
        profiles.append(profile);
    }
    return profiles;
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

QString ProfileManager::getActivePrinter()
{
    return activePrinterId;
}

PrinterProfile *ProfileManager::getActivePrinterProfile() const
{
    //QMap<QString, PrinterProfile*> systemPrinters; //read only
    //QMap<QString, PrinterProfile*> userPrinters; //editable & savable

    //TODO: find active printer profile in one of the QMaps above without adding any
    if (systemPrinters.contains(activePrinterId)) {
        return systemPrinters.value(activePrinterId);
    }

    if (userPrinters.contains(activePrinterId)) {
        return userPrinters.value(activePrinterId);
    }

    qWarning() << "[PROFILE MANAGER] Active printer not found:" << activePrinterId;
    return nullptr;
}

void ProfileManager::addUserPrinter(PrinterProfile *profile)
{
    if (!profile) {
        return;
    }

    //force isSystem = false
    profile->setIsSystem(false);

    QString id = profile->getId();

    // Ensure unique ID
    int suffix = 0;
    QString newId = generateUniquePrinterId(id, &suffix);
    profile->setId(newId);

    // Update display name: add suffix if necessary
    if (suffix > 0) {
        profile->setDisplayName(
            profile->getDisplayName() + " (" + QString::number(suffix) + ")");
    }

    userPrinters.insert(newId, profile);

    // Write JSON to user directory
    savePrinterProfile(profile);

    activePrinterId = newId;

    emit printersChanged();
    emit activePrinterChanged(newId);
}

void ProfileManager::updateUserPrinter(PrinterProfile *profile)
{
    if (!profile) {
        return;
    }

    const QString id = profile->getId();

    // Ensure unique ID
    if (!userPrinters.contains(id)) {
        qWarning() << "[PROFILE MANAGER] Tried to update non-user printer:" << id;
        return;
    }

    userPrinters[id] = profile;

    // Overwrite JSON
    savePrinterProfile(profile);

    if (id == activePrinterId) {
        emit activePrinterChanged(id);
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

void ProfileManager::deleteUserPrinter(const QString &id)
{
    if (!userPrinters.contains(id)) {
        qDebug() << "[PROFILE MANAGER] Tried to delete non-existent printer:" << id;
        return;
    }

    // Delete JSON file
    QString path = getUserPrinterDir() + "/" + id + ".json";
    if (QFile::exists(path)) {
        QFile::remove(path);
    }

    // Remove from map
    userPrinters.remove(id);

    emit printersChanged();

    // If the deleted printer was active, set a new active printer
    if (activePrinterId == id) {
        if (!userPrinters.isEmpty()) {
            setActivePrinter(userPrinters.firstKey());
        } else if (!systemPrinters.isEmpty()) {
            setActivePrinter(systemPrinters.firstKey());
        } else {
            activePrinterId.clear();
        }
    }
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
        file.close();

        if (!doc.isObject())
            continue;

        PrinterProfile* profile = PrinterProfile::fromJson(doc.object(), system);
        (system ? systemPrinters : userPrinters).insert(profile->getId(), profile);

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
        newId = baseId + QString::number(suffix);
        suffix++;
    } while (systemPrinters.contains(newId) || userPrinters.contains(newId));

    if (outSuffix) *outSuffix = suffix - 1;
    return newId;
}

