//
// Created by peter on 4/3/22.
// Sequential colormap: Blues, Greens, Greys, Purples, Reds, BuPu, GnBu, PuBu, PuBuGn, PuRd, RdPu, OrRd,
// RdOrYl, YlGn, YlGnBu, YlOrRd, BrBG, PuOr, RdBu, RdGy, RdYlBu, RdYlGn, PiYG, PRGn

#ifndef SEQCOLORBASE_H
#define SEQCOLORBASE_H
#include <iostream>
#include <Eigen/Core>
#include <map>

namespace colormap {
    namespace sequential {
        const Eigen::MatrixXi Blues = [] {
            Eigen::MatrixXi Blues(9, 3);
            Blues << 247, 251, 255,
                    222, 235, 247,
                    198, 219, 239,
                    158, 202, 225,
                    107, 174, 214,
                    66, 146, 198,
                    33, 113, 181,
                    8,  81, 156,
                    8,  48, 107;
            return Blues;
        }();

        const Eigen::MatrixXi Greens = [] {
            Eigen::MatrixXi Greens(9, 3);
            Greens << 247, 252, 245,
                    229, 245, 224,
                    199, 233, 192,
                    161, 217, 155,
                    116, 196, 118,
                    65, 171,  93,
                    35, 139,  69,
                    0, 109,  44,
                    0,  68,  27;
            return Greens;
        }();

        const Eigen::MatrixXi Greys = [] {
            Eigen::MatrixXi Greys(2, 3);
            Greys << 255, 255, 255,
                    0,   0,   0;
            return Greys;
        }();

        const Eigen::MatrixXi Oranges = []{
            Eigen::MatrixXi Oranges(9, 3);
            Oranges << 255, 245, 235,
                    254, 230, 206,
                    253, 208, 162,
                    253, 174, 107,
                    253, 141,  60,
                    241, 105,  19,
                    217,  72,   1,
                    166,  54,   3,
                    127,  39,   4;
            return Oranges;
        }();

        const Eigen::MatrixXi Purples = []{
            Eigen::MatrixXi Purples(9, 3);
            Purples << 252, 251, 253,
                    239, 237, 245,
                    218, 218, 235,
                    188, 189, 220,
                    158, 154, 200,
                    128, 125, 186,
                    106,  81, 163,
                    84,  39, 143,
                    63,   0, 125;
            return Purples;
        }();

        const Eigen::MatrixXi Reds = []{
            Eigen::MatrixXi Reds(9, 3);
            Reds << 255, 245, 240,
                    254, 224, 210,
                    252, 187, 161,
                    252, 146, 114,
                    251, 106,  74,
                    239,  59,  44,
                    203,  24,  29,
                    165,  15,  21,
                    103,   0,  13;
            return Reds;
        }();

        const Eigen::MatrixXi BuPu = []{
            Eigen::MatrixXi BuPu(9, 3);
            BuPu << 247, 252, 253,
                    224, 236, 244,
                    191, 211, 230,
                    158, 188, 218,
                    140, 150, 198,
                    140, 107, 177,
                    136,  65, 157,
                    129,  15, 124,
                    77,   0,  75;
            return BuPu;
        }();

        const Eigen::MatrixXi GnBu = []{
            Eigen::MatrixXi GnBu(9, 3);
            GnBu << 247, 252, 240,
                    224, 243, 219,
                    204, 235, 197,
                    168, 221, 181,
                    123, 204, 196,
                    78, 179, 211,
                    43, 140, 190,
                    8, 104, 172,
                    8,  64, 129;
            return GnBu;
        }();

        const Eigen::MatrixXi PuBu = []{
            Eigen::MatrixXi PuBu(9, 3);
            PuBu << 255, 247, 251,
                    236, 231, 242,
                    208, 209, 230,
                    166, 189, 219,
                    116, 169, 207,
                    54, 144, 192,
                    5, 112, 176,
                    4,  90, 141,
                    2,  56,  88;
            return PuBu;
        }();

        const Eigen::MatrixXi PuBuGn = []{
            Eigen::MatrixXi PuBuGn(9, 3);
            PuBuGn << 255, 247, 251,
                    236, 226, 240,
                    208, 209, 230,
                    166, 189, 219,
                    103, 169, 207,
                    54, 144, 192,
                    2, 129, 138,
                    1, 108,  89,
                    1,  70,  54;
            return PuBuGn;
        }();

        const Eigen::MatrixXi PuRd = []{
            Eigen::MatrixXi PuRd(9, 3);
            PuRd << 247, 244, 249,
                    231, 225, 239,
                    212, 185, 218,
                    201, 148, 199,
                    223, 101, 176,
                    231,  41, 138,
                    206,  18,  86,
                    152,   0,  67,
                    103,   0,  31;
            return PuRd;
        }();

        const Eigen::MatrixXi RdPu = []{
            Eigen::MatrixXi RdPu(9, 3);
            RdPu << 255, 247, 243,
                    253, 224, 221,
                    252, 197, 192,
                    250, 159, 181,
                    247, 104, 161,
                    221,  52, 151,
                    174,   1, 126,
                    122,   1, 119,
                    73,   0, 106;
            return RdPu;
        }();

        const Eigen::MatrixXi OrRd = []{
            Eigen::MatrixXi OrRd(9, 3);
            OrRd << 255, 247, 236,
                    254, 232, 200,
                    253, 212, 158,
                    253, 187, 132,
                    252, 141,  89,
                    239, 101,  72,
                    215,  48,  31,
                    179,   0,   0,
                    127,   0,   0;
            return OrRd;
        }();

        const Eigen::MatrixXi RdOrYl = []{
            Eigen::MatrixXi RdOrYl(9, 3);
            RdOrYl << 128, 0  , 38 ,
                    189, 0  , 38 ,
                    227, 26 , 28 ,
                    252, 78 , 42 ,
                    253, 141, 60 ,
                    254, 178, 76 ,
                    254, 217, 118,
                    255, 237, 160,
                    255, 255, 204;
            return RdOrYl;
        }();

        const Eigen::MatrixXi YlGn = []{
            Eigen::MatrixXi YlGn(9, 3);
            YlGn << 255, 255, 229,
                    247, 252, 185,
                    217, 240, 163,
                    173, 221, 142,
                    120, 198, 121,
                    65, 171,  93,
                    35, 132,  67,
                    0, 104,  55,
                    0,  69,  41;
            return YlGn;
        }();

        const Eigen::MatrixXi YlGnBu = []{
            Eigen::MatrixXi YlGnBu(9, 3);
            YlGnBu << 255, 255, 217,
                    237, 248, 177,
                    199, 233, 180,
                    127, 205, 187,
                    65, 182, 196,
                    29, 145, 192,
                    34,  94, 168,
                    37,  52, 148,
                    8,  29,  88;
            return YlGnBu;
        }();

        const Eigen::MatrixXi YlOrRd = []{
            Eigen::MatrixXi YlOrRd(9, 3);
            YlOrRd << 255, 255, 204,
                    255, 237, 160,
                    254, 217, 118,
                    254, 178,  76,
                    253, 141,  60,
                    252,  78,  42,
                    227,  26,  28,
                    189,   0,  38,
                    128,   0,  38;
            return YlOrRd;
        }();

        const Eigen::MatrixXi BrBG = []{
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

        const Eigen::MatrixXi PuOr = []{
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

        const Eigen::MatrixXi RdBu = []{
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

        const Eigen::MatrixXi RdGy = []{
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

        const Eigen::MatrixXi RdYlBu = []{
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

        const Eigen::MatrixXi RdYlGn = []{
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

        const Eigen::MatrixXi PiYG = []{
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

        const Eigen::MatrixXi PRGn = []{
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
                {"Blues", Blues},
                {"Greens", Greens},
                {"Greys", Greys},
                {"Oranges", Oranges},
                {"Purples", Purples},
                {"Reds", Reds},
                {"BuPu", BuPu},
                {"GnBu", GnBu},
                {"PuBu", PuBu},
                {"PuBuGn", PuBuGn},
                {"PuRd", PuRd},
                {"RdPu", RdPu},
                {"OrRd", OrRd},
                {"RdOrYl", RdOrYl},
                {"YlGn", YlGn},
                {"YlGnBu", YlGnBu},
                {"YlOrRd", YlOrRd},
                {"BrBG", BrBG},
                {"PuOr", PuOr},
                {"RdBu", RdBu},
                {"RdGy", RdGy},
                {"RdYlBu", RdYlBu},
                {"RdYlGn", RdYlGn},
                {"PiYG", PiYG},
                {"PRGn", PRGn}
        };
    }
}
#endif //SEQCOLORBASE_H
