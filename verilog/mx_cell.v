///////////////////////////////////////
//
//
// this is the mx_cell, which consists of 8 macs
module mx_cell #(
    parameter NUM_COE_ARRAY      = 16,
    parameter NUM_COMBINED_TERMS = 8 ,
    parameter NUM_BIT_EXPONENT   = 3
) (
    input                                            clk            ,
    input                                            reset          ,
    input  [                    8*NUM_COE_ARRAY-1:0] accumulation_in,
    input  [                                    3:0] data_terms     ,
    input  [                                    4:0] group_size     ,
    input  [                                    6:0] group_budget   ,
    input  [NUM_BIT_EXPONENT*NUM_COMBINED_TERMS-1:0] dataflow_in    ,
    input  [                 NUM_COMBINED_TERMS-1:0] sign_in        ,
    output [  NUM_BIT_EXPONENT*NUM_BIT_EXPONENT-1:0] dataflow_out   ,
    output [                 NUM_COMBINED_TERMS-1:0] sign_out       ,
    output [                    8*NUM_COE_ARRAY-1:0] result
);


    function [31:0] clog2 (input [31:0] x);
        reg [31:0] x_tmp;
        begin
            x_tmp = x-1;
            for(clog2=0; x_tmp>0; clog2=clog2+1) begin
                x_tmp = x_tmp >> 1;
            end
        end
    endfunction

    wire [                                    2:0] start           ;
    reg  [                                    2:0] counter         ;
    reg  [NUM_COMBINED_TERMS*NUM_BIT_EXPONENT-1:0] dataflow_in_reg ;
    reg  [                 NUM_COMBINED_TERMS-1:0] sign_in_reg     ;
    reg  [NUM_COMBINED_TERMS*NUM_BIT_EXPONENT-1:0] dataflow_out_reg;
    reg  [                 NUM_COMBINED_TERMS-1:0] sign_out_reg    ;

    assign dataflow_out = dataflow_out_reg;
    assign sign_out     = sign_out_reg;

// the declration of the start signal
    demux1to8(.clk(clk),.reset(reset),.sel(counter),.Data_out(start));

    always @(posedge clk) begin
        if(reset) begin
            {dataflow_in_reg,dataflow_out_reg} <= 'b0;
            counter <= 4'b0;
            {sign_in_reg,sign_out_reg} <= 'b0;
        end else begin
            counter          <= counter + 1'b1;
            dataflow_in_reg  <= dataflow_in;
            sign_in_reg      <= sign_in;
            dataflow_out_reg <= dataflow_in_reg;
            sign_out_reg     <= sign_in_reg;
        end
    end

    genvar i;
    generate
        for (i=0; i<8; i=i+1) begin
            begin : MAC_BLOCK
                mac #(.NUM_COE_ARRAY(NUM_COE_ARRAY),.NUM_COMBINED_TERMS(NUM_COMBINED_TERMS),.NUM_BIT_EXPONENT(NUM_BIT_EXPONENT))
                    pe (
                        .clk         (clk                        ),
                        .reset       (reset                      ),
                        .accumulation(accumulation_in[(i+1)*NUM_COE_ARRAY-1:i*NUM_COE_ARRAY]         ),
                        .input_exponent  (dataflow_in_reg       ),    // change here to count for the late start signal
                        .start_shift    (start[i]               ),
                        .sign_ctrl   (sign_in_reg               ),
                        .data_terms        (data_terms                                                      ),
                        .group_size        (group_size                                                      ),
                        .group_budget      (group_budget                                                    ),
                        .output_result      (result[(i+1)*NUM_COE_ARRAY-1:i*NUM_COE_ARRAY] )
                    );
            end
        end
    endgenerate
endmodule







