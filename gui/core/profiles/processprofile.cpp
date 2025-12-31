#include "processprofile.h"

ProcessProfile::ProcessProfile(const QString& id, bool system)
    : id(id), displayName(id), isSystem(system) {}

ProcessProfile* ProcessProfile::fromJson(const QJsonObject& obj, bool system)
{
    auto* p = new ProcessProfile(obj["id"].toString(), system);
    p->displayName = obj["name"].toString();

    p->layerHeight = obj["quality"].toObject()["layerHeight"].toDouble();
    p->wallLoops = obj["strength"].toObject()["wallLoops"].toInt();
    p->infillDensity = obj["strength"].toObject()["infillDensity"].toInt();
    p->supports = obj["support"].toObject()["enabled"].toBool();

    return p;
}

QJsonObject ProcessProfile::toJson() const
{
    return {
        {"id", id},
        {"name", displayName},
        {"quality", QJsonObject{{"layerHeight", layerHeight}}},
        {"strength", QJsonObject{
                         {"wallLoops", wallLoops},
                         {"infillDensity", infillDensity}
                     }},
        {"support", QJsonObject{{"enabled", supports}}}
    };
}

ProcessProfile* ProcessProfile::clone() const
{
    auto* c = new ProcessProfile(id, isSystem);
    *c = *this;
    return c;
}
