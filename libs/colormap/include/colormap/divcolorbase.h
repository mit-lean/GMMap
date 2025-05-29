//
// Created by peter on 4/3/22.
// Diverging colormaps

#ifndef DIVCOLORBASE_H
#define DIVCOLORBASE_H
#include <iostream>
#include <Eigen/Core>
#include <map>

namespace colormap {
    namespace div {
        const Eigen::MatrixXi BrBG = [] {
            Eigen::MatrixXi BrBG(11, 3);
            BrBG <<  84,  48,   5,
                    140,  81,  10,
                    191, 129,  45,
                    223, 194, 125,
                    246, 232, 195,
                    245, 245, 245,
                    199, 234, 229,
                    128, 205, 193,
                     53, 151, 143,
                      1, 102,  94,
                      0,  60,  48;
            return BrBG;
        }();

        const Eigen::MatrixXi PuOr = [] {
            Eigen::MatrixXi PuOr(11, 3);
            PuOr << 127,  59,   8,
                    179,  88,   6,
                    224, 130,  20,
                    253, 184,  99,
                    254, 224, 182,
                    247, 247, 247,
                    216, 218, 235,
                    178, 171, 210,
                    128, 115, 172,
                     84,  39, 136,
                     45,   0,  75;
            return PuOr;
        }();

        const Eigen::MatrixXi RdBu = [] {
            Eigen::MatrixXi RdBu(11, 3);
            RdBu << 103,   0,  31,
                    178,  24,  43,
                    214,  96,  77,
                    244, 165, 130,
                    253, 219, 199,
                    247, 247, 247,
                    209, 229, 240,
                    146, 197, 222,
                     67, 147, 195,
                     33, 102, 172,
                      5,  48,  97;
            return RdBu;
        }();

        const Eigen::MatrixXi RdGy = [] {
            Eigen::MatrixXi RdGy(11, 3);
            RdGy << 103,   0,  31,
                    178,  24,  43,
                    214,  96,  77,
                    244, 165, 130,
                    253, 219, 199,
                    255, 255, 255,
                    224, 224, 224,
                    186, 186, 186,
                    135, 135, 135,
                     77,  77,  77,
                     26,  26,  26;
            return RdGy;
        }();

        const Eigen::MatrixXi RdYlBu = [] {
            Eigen::MatrixXi RdYlBu(11, 3);
            RdYlBu << 165,   0,  38,
                        215,  48,  39,
                        244, 109,  67,
                        253, 174,  97,
                        254, 224, 144,
                        255, 255, 191,
                        224, 243, 248,
                        171, 217, 233,
                        116, 173, 209,
                         69, 117, 180,
                         49,  54, 149;
            return RdYlBu;
        }();

        const Eigen::MatrixXi BuYlRd = RdYlBu.colwise().reverse();

        const Eigen::MatrixXi RdYlGn = [] {
            Eigen::MatrixXi RdYlGn(11, 3);
            RdYlGn << 165,   0,  38,
                    215,  48,  39,
                    244, 109,  67,
                    253, 174,  97,
                    254, 224, 139,
                    255, 255, 191,
                    217, 239, 139,
                    166, 217, 106,
                    102, 189,  99,
                     26, 152,  80,
                      0, 104,  55;
            return RdYlGn;
        }();

        const Eigen::MatrixXi PiYG = [] {
            Eigen::MatrixXi PiYG(11, 3);
            PiYG << 142,   1,  82,
                    197,  27, 125,
                    222, 119, 174,
                    241, 182, 218,
                    253, 224, 239,
                    247, 247, 247,
                    230, 245, 208,
                    184, 225, 134,
                    127, 188,  65,
                     77, 146,  33,
                     39, 100,  25;
            return PiYG;
        }();

        const Eigen::MatrixXi PRGn = [] {
            Eigen::MatrixXi PRGn(11, 3);
            PRGn <<  64,   0,  75,
                    118,  42, 131,
                    153, 112, 171,
                    194, 165, 207,
                    231, 212, 232,
                    247, 247, 247,
                    217, 240, 211,
                    166, 219, 160,
                     90, 174,  97,
                     27, 120,  55,
                      0,  68,  27;
            return PRGn;
        }();

        const std::map<std::string, Eigen::MatrixXi> colorbase = {
                {"BrBG", BrBG},
                {"PuOr", PuOr},
                {"RdBu", RdBu},
                {"RdGy", RdGy},
                {"RdYlBu", RdYlBu},
                {"BuYlRd", BuYlRd},
                {"RdYlGn", RdYlGn},
                {"PiYG", PiYG},
                {"PRGn", PRGn}
        };
    }
}

#endif //DIVCOLORBASE_H
